import sys
sys.path.append(".")

from JOBParser import TargetTable,FromTable,Comparison
# max_column_in_table = 15
import torch
import torch
import torch.nn as nn
from itertools import count
import numpy as np
import threading
from PGUtils import pgrunner
from JOBParser import TargetTable,FromTable,Comparison
from ImportantConfig import Config
config = Config()
def zero_hc(input_dim = 1):
    return torch.zeros(input_dim,config.hidden_size,device = config.device),torch.zeros(input_dim,config.hidden_size,device = config.device)
# Module-global column id map, mutated by getColumnId on every parse. Concurrent
# optimize threads parse in parallel, so the check-then-assign (`if not in:
# column_id[col] = len(column_id)`) must be serialized or two threads could
# assign the same new column different ids (inconsistent features ↔ weights).
# The wrapper's mtime-reload also replaces this dict under the same lock.
column_id = {}
_column_id_lock = threading.Lock()
def getColumnId(column):
    with _column_id_lock:
        if not column in column_id:
            column_id[column] = len(column_id)
        return column_id[column]
class Sql2Vec:
    def __init__(self,):
        pass
    def to_vec(self,sql):
        # All intermediates are LOCAL (not self.X): to_vec runs concurrently
        # across optimize threads on this shared Sql2Vec instance, and as
        # instance attrs two different-query calls would clobber each other's
        # join_list / aliasname2id / join_matrix — findBestHint would then read
        # another query's joins and the MCTS would dead-end. Returns the data
        # findBestHint needs so it doesn't read instance attrs either.
        from psqlparse import parse_dict
        parse_result = parse_dict(sql)[0]["SelectStmt"]
        target_table_list = [TargetTable(x["ResTarget"]) for x in parse_result["targetList"]]
        from_table_list = [FromTable(x["RangeVar"]) for x in parse_result["fromClause"]]
        if len(from_table_list)<2:
            return None

        id2aliasname = config.id2aliasname
        aliasname2id = config.aliasname2id
        aliasname2fullname = {}

        join_list = set()
        aliasnames_root_set = set([x.getAliasName() for x in from_table_list])

        alias_selectivity = np.asarray([0]*len(id2aliasname),dtype = np.float)
        aliasname2fromtable = {}
        for table in from_table_list:
            aliasname2fromtable[table.getAliasName()] = table
            aliasname2fullname[table.getAliasName()] = table.getFullName()

        aliasnames = set(aliasname2fromtable.keys())
        comparison_list =[Comparison(x) for x in parse_result["whereClause"]["BoolExpr"]["args"]]
        join_matrix = np.zeros((len(id2aliasname),len(id2aliasname)),dtype = np.float)
        count_selectivity = np.asarray([0]*config.max_column,dtype = np.float)
        has_predicate = set()
        join_list_with_predicate = set()
        for comparison in comparison_list:
            if len(comparison.aliasname_list) == 2:
                left_aliasname = comparison.aliasname_list[0]
                right_aliasname = comparison.aliasname_list[1]
                idx0 = aliasname2id[left_aliasname]
                idx1 = aliasname2id[right_aliasname]
                if idx0<idx1:
                    join_list.add((left_aliasname,right_aliasname))
                else:
                    join_list.add((right_aliasname,left_aliasname))
                join_matrix[idx0][idx1] = 1
                join_matrix[idx1][idx0] = 1
            else:
                left_aliasname = comparison.aliasname_list[0]
                alias_selectivity[aliasname2id[left_aliasname]] = alias_selectivity[aliasname2id[left_aliasname]]+pgrunner.getSelectivity(str(aliasname2fromtable[comparison.aliasname_list[0]]),str(comparison))
                has_predicate.add(left_aliasname)
                count_selectivity[getColumnId(comparison.column)] = count_selectivity[getColumnId(comparison.column)]+pgrunner.getSelectivity(str(aliasname2fromtable[comparison.aliasname_list[0]]),str(comparison))
        for ajoin in join_list:
            if ajoin[0] in has_predicate or ajoin[1] in has_predicate :
                join_list_with_predicate.add(ajoin)
        if config.max_column==40:
            sql_vec = np.concatenate((join_matrix.flatten(),alias_selectivity))
        else:
            sql_vec = np.concatenate((join_matrix.flatten(),count_selectivity))
        return sql_vec, aliasnames_root_set, id2aliasname, aliasname2id, join_list, join_list_with_predicate

JOIN_TYPES = ["Nested Loop", "Hash Join", "Merge Join"]
LEAF_TYPES = ["Seq Scan", "Index Scan", "Index Only Scan", "Bitmap Index Scan"]
ALL_TYPES = JOIN_TYPES + LEAF_TYPES

class ValueExtractor:
    def __init__(self,offset=config.offset,max_value = 20):
        self.offset = offset
        self.max_value = max_value
    # def encode(self,v):
    #     return np.log(self.offset+v)/np.log(2)/self.max_value
    # def decode(self,v):
    #     # v=-(v*v<0)
    #     return np.exp(v*self.max_value*np.log(2))#-self.offset
    def encode(self,v):
        return int(np.log(2+v)/np.log(config.max_time_out)*200)/200.
        return int(np.log(self.offset+v)/np.log(config.max_time_out)*200)/200.
    def decode(self,v):
        # v=-(v*v<0)
        # return np.exp(v/2*np.log(config.max_time_out))#-self.offset
        return np.exp(v*np.log(config.max_time_out))#-self.offset
    def cost_encode(self,v,min_cost,max_cost):
        return (v-min_cost)/(max_cost-min_cost)
    def cost_decode(self,v,min_cost,max_cost):
        return (max_cost-min_cost)*v+min_cost
    def latency_encode(self,v,min_latency,max_latency):
        return (v-min_latency)/(max_latency-min_latency)
    def latency_decode(self,v,min_latency,max_latency):
        return (max_latency-min_latency)*v+min_latency
    def rows_encode(self,v,min_cost,max_cost):
        return (v-min_cost)/(max_cost-min_cost)
    def rows_decode(self,v,min_cost,max_cost):
        return (max_cost-min_cost)*v+min_cost
value_extractor = ValueExtractor()
def get_plan_stats(data):
    return [value_extractor.encode(data["Total Cost"]),value_extractor.encode(data["Plan Rows"])]

class TreeBuilderError(Exception):
    def __init__(self, msg):
        self.__msg = msg

def is_join(node):
    return node["Node Type"] in JOIN_TYPES

def is_scan(node):
    return node["Node Type"] in LEAF_TYPES

# fasttext
class PredicateEncode:
    def __init__(self,):
        pass
    def stringEncoder(self,string_predicate):
        return torch.tensor([0,1]+[0]*config.hidden_size,device = config.device).float()
        pass
    def floatEncoder(self,float1,float2):
        return torch.tensor([float1,float2]+[0]*config.hidden_size,device = config.device).float()
        pass
class TreeBuilder:
    def __init__(self):
        self.__stats = get_plan_stats
        self.id2aliasname = config.id2aliasname
        self.aliasname2id = config.aliasname2id
        
    def __relation_name(self, node):
        if "Relation Name" in node:
            return node["Relation Name"]

        if node["Node Type"] == "Bitmap Index Scan":
            # find the first (longest) relation name that appears in the index name
            name_key = "Index Name" if "Index Name" in node else "Relation Name"
            if name_key not in node:
                print(node)
                raise TreeBuilderError("Bitmap operator did not have an index name or a relation name")
            for rel in self.__relations:
                if rel in node[name_key]:
                    return rel

            raise TreeBuilderError("Could not find relation name for bitmap index scan")

        raise TreeBuilderError("Cannot extract relation type from node")
    def __alias_name(self, node):
        if "Alias" in node:
            return np.asarray([self.aliasname2id[node["Alias"]]])

        if node["Node Type"] == "Bitmap Index Scan":
            # find the first (longest) relation name that appears in the index name
            name_key = "Index Cond" #if "Index Cond" in node else "Relation Name"
            if name_key not in node:
                print(node)
                raise TreeBuilderError("Bitmap operator did not have an index name or a relation name")
            for rel in self.aliasname2id:
                if rel+'.' in node[name_key]:
                    return np.asarray([-1])
                    return np.asarray([self.aliasname2id[rel]])

        #     raise TreeBuilderError("Could not find relation name for bitmap index scan")
        print(node)
        raise TreeBuilderError("Cannot extract Alias type from node")
                
    def __featurize_join(self, node):
        assert is_join(node)
        # return [node["Node Type"],self.__stats(node),0,0]
        arr = np.zeros(len(ALL_TYPES))
        arr[ALL_TYPES.index(node["Node Type"])] = 1
        feature = np.concatenate((arr, self.__stats(node)))
        feature = torch.tensor(feature,device = config.device,dtype = torch.float32).reshape(-1,config.input_size)
        return feature

    def __featurize_scan(self, node):
        assert is_scan(node)
        # return [node["Node Type"],self.__stats(node),self.__alias_name(node)]
        arr = np.zeros(len(ALL_TYPES))
        arr[ALL_TYPES.index(node["Node Type"])] = 1
        feature = np.concatenate((arr, self.__stats(node)))
        feature = torch.tensor(feature,device = config.device,dtype = torch.float32).reshape(-1,config.input_size)
        return (feature,
                torch.tensor(self.__alias_name(node),device = config.device,dtype = torch.long))

    def plan_to_feature_tree(self, plan):
        
        
        # children = plan["Plans"] if "Plans" in plan else []
        if "Plan" in plan:
            plan = plan["Plan"]
        children = plan["Plan"] if "Plan" in plan else (plan["Plans"] if "Plans" in plan else [])
        if len(children) == 1:
            child_value = self.plan_to_feature_tree(children[0])
            if "Alias" in plan and plan["Node Type"]=='Bitmap Heap Scan':
                alias_idx_np = np.asarray([self.aliasname2id[plan["Alias"]]])
                if isinstance(child_value[1],tuple):
                    raise TreeBuilderError("Node wasn't transparent, a join, or a scan: " + str(plan))
                return (child_value[0],torch.tensor(alias_idx_np,device = config.device,dtype = torch.long))
            return child_value
        # print(plan)
        if is_join(plan):
            assert len(children) == 2
            my_vec = self.__featurize_join(plan)
            left = self.plan_to_feature_tree(children[0])
            right = self.plan_to_feature_tree(children[1])
            # print('is_join',my_vec)
            return (my_vec, left, right)

        if is_scan(plan):
            assert not children
            # print(plan)
            s = self.__featurize_scan(plan)
            # print('is_scan',s)
            return s

        raise TreeBuilderError("Node wasn't transparent, a join, or a scan: " + str(plan))


        
                
                
