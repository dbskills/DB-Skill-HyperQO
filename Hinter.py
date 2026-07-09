from ImportantConfig import Config
from math import e
from PGUtils import pgrunner
import torch
from KNN import KNN
import time


def formatFloat(t):
    try:
        return " ".join(["{:.4f}".format(x) for x in t])
    except:
        return " ".join(["{:.4f}".format(x) for x in [t]])
config = Config()

class Timer:
    def __init__(self,):
        from time import time
        self.timer = time
        self.startTime = {}
    def reset(self,s):
        self.startTime[s] = self.timer()
    def record(self,s):
        return self.timer()-self.startTime[s]
timer = Timer()

class Hinter:
    def __init__(self,model,sql2vec,value_extractor,mcts_searcher=None,optimize_only=False):
        self.model = model #Net.TreeNet
        self.sql2vec = sql2vec#
        self.value_extractor = value_extractor
        self.optimize_only = optimize_only
        self.knn = KNN(10)
        self.mcts_searcher = mcts_searcher
        self.hinter_times = 0

    def findBestHint(self,plan_json_PG,alias,sql_vec,sql,id2aliasname,aliasname2id,join_list,join_list_with_predicate,mcts_time_list,MHPE_time_list):
        alias_id = [aliasname2id[a] for a in alias]
        timer.reset('mcts_time_list')
        id_joins_with_predicate = [(aliasname2id[p[0]],aliasname2id[p[1]]) for p in join_list_with_predicate]
        id_joins = [(aliasname2id[p[0]],aliasname2id[p[1]]) for p in join_list]
        leading_length = config.leading_length
        if leading_length==-1:
            leading_length = len(alias)
        if leading_length>len(alias):
            leading_length = len(alias)
        join_list_with_predicate_hint = self.mcts_searcher.findCanHints(40,len(alias),sql_vec,id_joins,id_joins_with_predicate,alias_id,depth=leading_length)
        mcts_time_list.append(timer.record('mcts_time_list'))

        leading_list = []
        plan_jsons = []
        leadings_utility_list = []
        for join in join_list_with_predicate_hint:
            leading_list.append('/*+Leading('+" ".join([id2aliasname[x] for x in join[0][:leading_length]])+')*/')
            leadings_utility_list.append(join[1])
            ##To do: parallel planning
            plan_jsons.append(pgrunner.getCostPlanJson(leading_list[-1]+sql))
        plan_jsons.extend([plan_json_PG])
        timer.reset('MHPE_time_list')
        plan_times = self.predictWithUncertaintyBatch(plan_jsons=plan_jsons,sql_vec = sql_vec)
        MHPE_time_list.append(timer.record('MHPE_time_list'))
        chosen_leading_pair = sorted(zip(plan_times[:config.max_hint_num],leading_list,leadings_utility_list),key = lambda x:x[0][0]+self.knn.kNeightboursSample(x[0]))[0]
        # plan_times[-1] is PG's prediction (plan_json_PG is appended last at the
        # extend above). Returned so hinterRun can reuse it instead of a separate
        # standalone forward over [PG] (which would recompute the same value).
        return chosen_leading_pair, plan_times[-1]

    def hinterRun(self,sql):
        self.hinter_times += 1
        # Per-call locals (NOT instance attrs). These lists are appended
        # throughout hinterRun and read via [-1] at the return; as instance
        # attrs they would race between concurrent optimize calls (and [-1]
        # could return another call's value). As locals, hinterRun is
        # thread-safe w.r.t. this bookkeeping. The training block (skipped in
        # optimize-only mode) reads `samples_plan_with_time` — the local.
        pg_planningtime_list = []
        pg_runningtime_list = []
        mcts_time_list = []
        hinter_planningtime_list = []
        MHPE_time_list = []
        hinter_runtime_list = []
        chosen_plan = []
        hinter_time_list = []
        samples_plan_with_time = []
        plan_json_PG = pgrunner.getCostPlanJson(sql)
        mask = (torch.rand(1,config.head_num,device = config.device)<0.9).long()

        if self.optimize_only:
            # In optimize-only mode, skip EXPLAIN ANALYZE and actual execution.
            # Use cost estimates from getCostPlanJson instead.
            pg_plan_time = plan_json_PG.get('Planning Time', 0)
            pg_runtime = plan_json_PG['Plan']['Total Cost']
            pg_planningtime_list.append(pg_plan_time)
            pg_runningtime_list.append(pg_runtime)
        elif config.cost_test_for_debug:
            pg_runningtime_list.append(pgrunner.getCost(sql)[0])
            pg_planningtime_list.append(pgrunner.getCostPlanJson(sql)['Planning Time'])
        else:
            pg_runningtime_list.append(pgrunner.getAnalysePlanJson(sql)['Plan']['Actual Total Time'])
            pg_planningtime_list.append(pgrunner.getAnalysePlanJson(sql)['Planning Time'])

        sql_vec_alias = self.sql2vec.to_vec(sql)
        if sql_vec_alias is None:
            sql_vec, alias, id2aliasname, aliasname2id, join_list, join_list_with_predicate = None, [], {}, {}, set(), set()
        else:
            sql_vec, alias, id2aliasname, aliasname2id, join_list, join_list_with_predicate = sql_vec_alias
        # PG's (mean,variance,v2) is returned by findBestHint, which predicts it as
        # part of its single batched forward over [candidates + PG]. The SPINN
        # forward is deterministic (Dropout is defined but never called), so a
        # separate standalone forward over [PG] alone would be pure redundancy.
        chosen_leading_pair, pg_value = self.findBestHint(plan_json_PG=plan_json_PG,alias=alias,sql_vec = sql_vec,sql=sql,id2aliasname=id2aliasname,aliasname2id=aliasname2id,join_list=join_list,join_list_with_predicate=join_list_with_predicate,mcts_time_list=mcts_time_list,MHPE_time_list=MHPE_time_list)
        knn_plan = abs(self.knn.kNeightboursSample(pg_value))
        if chosen_leading_pair[0][0]<pg_value[0] and abs(knn_plan)<config.threshold and self.value_extractor.decode(pg_value[0])>100:
            from math import e
            max_time_out = min(int(self.value_extractor.decode(chosen_leading_pair[0][0])*3),config.max_time_out)
            if self.optimize_only:
                # In optimize-only mode, use cost estimate instead of EXPLAIN ANALYZE
                hinted_plan = pgrunner.getCostPlanJson(sql = chosen_leading_pair[1]+sql)
                leading_time_flag = (hinted_plan['Plan']['Total Cost'], False)
                hinter_runtime_list.append(leading_time_flag[0])
                hinter_planningtime_list.append(hinted_plan.get('Planning Time', 0))
                hinter_time_list.append([leading_time_flag[0]])
                chosen_plan.append([chosen_leading_pair[1]])
            elif config.cost_test_for_debug:
                leading_time_flag = pgrunner.getCost(sql = chosen_leading_pair[1]+sql)
                hinter_runtime_list.append(leading_time_flag[0])
                ##To do: parallel planning
                hinter_planningtime_list.append(pgrunner.getCostPlanJson(sql = chosen_leading_pair[1]+sql)['Planning Time'])
            else:
                plan_json  = pgrunner.getAnalysePlanJson(sql = chosen_leading_pair[1]+sql)
                leading_time_flag = (plan_json['Plan']['Actual Total Time'],plan_json['timeout'])
                hinter_runtime_list.append(leading_time_flag[0])
                ##To do: parallel planning
                hinter_planningtime_list.append(plan_json['Planning Time'])

            if not self.optimize_only:
                self.knn.insertAValue((chosen_leading_pair[0],self.value_extractor.encode(leading_time_flag[0])-chosen_leading_pair[0][0]))
            if config.cost_test_for_debug:
                samples_plan_with_time.append([pgrunner.getCostPlanJson(sql = chosen_leading_pair[1]+sql,timeout=max_time_out),leading_time_flag[0],mask])
            elif self.optimize_only:
                # Collect training sample with cost estimate (no EXPLAIN ANALYZE)
                hinted_plan = pgrunner.getCostPlanJson(sql = chosen_leading_pair[1]+sql,timeout=max_time_out)
                samples_plan_with_time.append([hinted_plan,leading_time_flag[0],mask])
            else:
                samples_plan_with_time.append([pgrunner.getCostPlanJson(sql = chosen_leading_pair[1]+sql,timeout=max_time_out),leading_time_flag[0],mask])
            if not self.optimize_only and leading_time_flag[1]:
                if config.cost_test_for_debug:
                    pg_time_flag = pgrunner.getCost(sql=sql)
                else:
                    pg_time_flag = pgrunner.getLatency(sql=sql,timeout = 300*1000)
                self.knn.insertAValue((pg_value,self.value_extractor.encode(pg_time_flag[0])-pg_value[0]))
                if samples_plan_with_time[0][1]>pg_time_flag[0]*1.8:
                    samples_plan_with_time[0][1] = pg_time_flag[0]*1.8
                    samples_plan_with_time.append([plan_json_PG,pg_time_flag[0],mask])
                else:
                    samples_plan_with_time[0] = [plan_json_PG,pg_time_flag[0],mask]

                hinter_time_list.append([max_time_out,pgrunner.getLatency(sql=sql,timeout = 300*1000)[0]])
                chosen_plan.append([chosen_leading_pair[1],'PG'])
            elif not self.optimize_only:
                hinter_time_list.append([leading_time_flag[0]])
                chosen_plan.append([chosen_leading_pair[1]])
        else:
            if self.optimize_only:
                # Use cost estimate instead of actual execution
                pg_time_flag = (plan_json_PG['Plan']['Total Cost'], 0)
                hinter_runtime_list.append(pg_time_flag[0])
                hinter_planningtime_list.append(plan_json_PG.get('Planning Time', 0))
            elif config.cost_test_for_debug:
                pg_time_flag = pgrunner.getCost(sql=sql)
                hinter_runtime_list.append(pg_time_flag[0])
                ##To do: parallel planning
                hinter_planningtime_list.append(pgrunner.getCostPlanJson(sql)['Planning Time'])
            else:
                pg_time_flag = pgrunner.getLatency(sql=sql,timeout = 300*1000)
                hinter_runtime_list.append(pg_time_flag[0])
                ##To do: parallel planning

                hinter_planningtime_list.append(pgrunner.getAnalysePlanJson(sql = sql)['Planning Time'])
            if not self.optimize_only:
                self.knn.insertAValue((pg_value,self.value_extractor.encode(pg_time_flag[0])-pg_value[0]))
            samples_plan_with_time.append([plan_json_PG,pg_time_flag[0],mask])
            hinter_time_list.append([pg_time_flag[0]])
            chosen_plan.append(['PG'])

        # Training: skip entirely in optimize-only mode
        if not self.optimize_only:
            for sample in samples_plan_with_time:
                target_value = self.value_extractor.encode(sample[1])
                self.model.train(plan_json = sample[0],sql_vec = sql_vec,target_value=target_value,mask = mask,is_train = True)
                self.mcts_searcher.train(tree_feature = self.model.tree_builder.plan_to_feature_tree(sample[0]),sql_vec = sql_vec,target_value = sample[1],alias_set=alias)

            if self.hinter_times<1000 or self.hinter_times%10==0:
                loss=  self.model.optimize()[0]
                loss1 = self.mcts_searcher.optimize()
                if self.hinter_times<1000:
                    loss=  self.model.optimize()[0]
                    loss1 = self.mcts_searcher.optimize()
                if loss>3:
                    loss=  self.model.optimize()[0]
                    loss1 = self.mcts_searcher.optimize()
                if loss>3:
                    loss=  self.model.optimize()[0]
                    loss1 = self.mcts_searcher.optimize()


        assert len(set([len(hinter_runtime_list),len(pg_runningtime_list),len(mcts_time_list),len(hinter_planningtime_list),len(MHPE_time_list),len(hinter_runtime_list),len(chosen_plan),len(hinter_time_list)]))==1
        return pg_planningtime_list[-1],pg_runningtime_list[-1],mcts_time_list[-1],hinter_planningtime_list[-1],MHPE_time_list[-1],hinter_runtime_list[-1],chosen_plan[-1],hinter_time_list[-1]

    def predictWithUncertaintyBatch(self,plan_jsons,sql_vec):
        sql_feature = self.model.value_network.sql_feature(sql_vec)
        import torchfold
        fold = torchfold.Fold(cuda=config.device.type=="cuda")
        res = []
        multi_list = []
        for plan_json in plan_jsons:
            tree_feature = self.model.tree_builder.plan_to_feature_tree(plan_json)
            multi_value = self.model.plan_to_value_fold(tree_feature=tree_feature,sql_feature = sql_feature,fold=fold)
            multi_list.append(multi_value)
        multi_value = fold.apply(self.model.value_network,[multi_list])[0]
        mean,variance  = self.model.mean_and_variance(multi_value=multi_value[:,:config.head_num])
        v2 = torch.exp(multi_value[:,config.head_num]*config.var_weight).data.reshape(-1)
        if isinstance(mean,float):
            mean_item = [mean]
        else:
            mean_item = [x.item()for x in mean]
        if isinstance(variance,float):
            variance_item = [variance]
        else:
            variance_item = [x.item()for x in variance]
        if isinstance(v2,float):
            v2_item = [v2]
        else:
            v2_item = [x.item()for x in v2]
        res = list(zip(mean_item,variance_item,v2_item))
        return res
