import copy
from src.simlator.Simulator import *
import time
import random
class GA_allocate:
    population = [] #[ [job_seq,mac_seq, fittness], ...]
    MUTATION_PROB = 0.05
    POP_SIZE = 100
    RANGE = 0
    NUM_OFFSPRING = 5
    SELECTION_PRESSURE = 3
    END = 0.9
    job_type_list = []
    generation = 0  # 현재 세대 수
    fixed_lot = 10
    job_list = ['P1', 'P2', 'P3', 'P4']
    warehouse_list = ['ALPHA', 'BETA']

    job_type = {
        "P1":90,
        'P2':90,
        'P3':100,
        'P4':100
    }
    wh_to_concap = {
        'ALPHA' : ['A1', 'B1', 'C1'],
        'BETA' : ['A2', 'B2', 'C2']
    }
    concap_true = {
        'A1': 90000,
        'B1': 60000,
        'C1': 30000,
        'A2': 60000,
        'B2': 120000,
        'C2': 60000
    }
    concap = {
        'A1': 90000,
        'B1': 60000,
        'C1': 30000,
        'A2': 60000,
        'B2': 120000,
        'C2': 60000
    }

    consume_rate ={
        'P1' : {'A1':500, 'A2':500, 'B1':400, 'B2':400, 'C1':300, 'C2':300},
        'P2' : {'A1':600, 'A2':600, 'C1':700, 'C2':700},
        'P3' : {'A1':400, 'A2':400, 'B1':450, 'B2':450},
        'P4' : { 'B1':900, 'B2':900}
    }
    @classmethod
    def fixed_seq(cls):
        job_seq = []
        j_list = ['P1' for _ in range(90)]
        job_seq.append(j_list)
        j_list = ['P2' for _ in range(82)]
        job_seq.append(j_list)
        j_list = ['P3' for _ in range(100)]
        job_seq.append(j_list)
        j_list = ['P4' for _ in range(100)]
        job_seq.append(j_list)
        j_list = ['P2' for _ in range(8)]
        job_seq.append(j_list)
        total_job = sum(job_seq, [])
        wh_seq = []
        wh = ['ALPHA' for _ in range(38)]
        wh_seq.append(wh)
        wh = ['BETA' for _ in range(108)]
        wh_seq.append(wh)
        wh = ['ALPHA' for _ in range(26+99)]
        wh_seq.append(wh)
        wh = ['BETA' for _ in range(101)]
        wh_seq.append(wh)
        wh = ['BETA' for _ in range(8)]
        wh_seq.append(wh)
        total_wh = sum(wh_seq, [])

        return [total_job, total_wh]
    @classmethod
    def _init_job_seq(cls):
        job_seq = []
        for job_type, qty in cls.job_type.items():
            j_list = [job_type for _ in range(qty)]
            job_seq.append(j_list)
        total_job = sum(job_seq , [])
        random.shuffle(total_job)
        return total_job

    @classmethod
    def _init_wh_seq(cls, total_job):
        wh_seq = []
        for i in total_job:
            coin = random.randint(0,len(cls.warehouse_list )-1)
            wh_seq.append(cls.warehouse_list[coin])
        return wh_seq
    @classmethod
    def get_fittness(cls, solution):
        th = 0
        bt_neck = 0
        for i in range(len(solution[0])):
            prod = solution[0][i]
            wh = solution[1][i]
            can_dsp = True
            for concap_group in cls.consume_rate[prod]:
                if concap_group not in cls.wh_to_concap[wh]:
                    continue
                consume_rate = cls.consume_rate[prod][concap_group]
                if cls.concap[concap_group] < consume_rate:
                    can_dsp = False
                    if bt_neck == 0:
                        bt_neck = i
                    break

            if can_dsp:
                th+=1
                for concap_group in cls.consume_rate[prod]:
                    if concap_group not in cls.wh_to_concap[wh]:
                        continue
                    consume_rate = cls.consume_rate[prod][concap_group]
                    cls.concap[concap_group] -= consume_rate

        cls.concap = copy.deepcopy(cls.concap_true)

        return th, bt_neck

    @classmethod
    def init_population(cls):
        for i in range(cls.POP_SIZE):
            job_seq = cls._init_job_seq()
            wh_seq = cls._init_wh_seq(job_seq)
            fittness, bt_neck = cls.get_fittness([job_seq, wh_seq] )
            cls.population.append([job_seq, wh_seq, fittness, bt_neck])
    @classmethod
    def crossover_operator_JCO(cls, mom, dad):
        dad_ch = copy.deepcopy(dad[0])
        mom_ch = copy.deepcopy(mom[0])
        mom_ch_flow = copy.deepcopy(mom[1])
        dad_ch_flow = copy.deepcopy(dad[1])
        job_seq = cls.job_list
        change_job_list = []
        for job in job_seq:
            coin = random.random()
            if coin < 0.5:
                change_job_list.append(job)

        for i in range(len(dad_ch)):
            if dad_ch[i] in change_job_list:
                dad_ch[i] = None
            else:
                index = mom_ch.index(dad_ch[i])
                mom_ch.pop(index)
                mom_ch_flow.pop(index)
                
        for i in range(len(dad_ch)):
            if dad_ch[i] == None:
                dad_ch[i] = mom_ch.pop(0)
                dad_ch_flow[i] = mom_ch_flow.pop(0)

        throughput,bt_neck = cls.get_fittness([dad_ch, dad_ch_flow])
        offspring = [dad_ch, dad_ch_flow, throughput,bt_neck]
        return offspring

    @classmethod
    def mutation_operator(cls, offspring):
        for i in range(len(offspring[0])):
            coin = random.random()
            if coin < 0.05:
                if offspring[1][i] == "ALPHA":
                    offspring[1][i] = "BETA"
                else:
                    offspring[1][i] = "ALPHA"

        throughput, bt_neck = cls.get_fittness([offspring[0], offspring[1]])
        offspring = [offspring[0], offspring[1], throughput, bt_neck]
        return offspring

    @classmethod
    def mutation_operator_bt_neck(cls, offspring):
        bt_neck = offspring[3]
        if offspring[1][bt_neck] == "ALPHA":
            offspring[1][bt_neck] = "BETA"
        else:
            offspring[1][bt_neck] = "ALPHA"

        throughput, bt_neck = cls.get_fittness([offspring[0], offspring[1]])
        offspring = [offspring[0], offspring[1], throughput, bt_neck]
        return offspring

    @classmethod
    def mutation_operator_seq(cls, offspring):
        a = random.randint(0, len(offspring[0])//2)
        b = random.randint(len(offspring[0])//2, len(offspring[0]))

        # 지정된 두 인덱스 사이의 값을 뒤집기
        subsequence = offspring[0][a:b + 1]  # 시작부터 끝까지의 부분 리스트를 가져옵니다.
        subsequence.reverse()  # 부분 리스트를 뒤집습니다.
        offspring[0][a:b + 1] = subsequence  # 원래 리스트에 뒤집은 부분 리스트를 다시 할당합니다.

        # 지정된 두 인덱스 사이의 값을 뒤집기
        subsequence = offspring[1][a:b + 1]  # 시작부터 끝까지의 부분 리스트를 가져옵니다.
        subsequence.reverse()  # 부분 리스트를 뒤집습니다.
        offspring[1][a:b + 1] = subsequence  # 원래 리스트에 뒤집은 부분 리스트를 다시 할당합니다.


        throughput,bt_neck = cls.get_fittness([offspring[0], offspring[1]])
        offspring = [offspring[0], offspring[1], throughput,bt_neck]
        return offspring

    @classmethod
    def sort_population(cls):
        cls.population.sort(key = lambda x: x[2] ,reverse=True)
    @classmethod
    def selection_operator(cls):
        #룰렛 휠
        inverse_fitness = [1 / x[2] for x in cls.population]
        chrom = random.choices(cls.population, weights=inverse_fitness, k=2)
        mom_ch = chrom[0]
        dad_ch = chrom[1]
        return mom_ch, dad_ch

    @classmethod
    def replacement_operator(cls, offsprings):
        result_population = []
        for i in range(cls.NUM_OFFSPRING):
            cls.population.pop()
        for i in range(cls.NUM_OFFSPRING):
            cls.population.append(offsprings[i])


    @classmethod
    def print_avg_fittness(cls):
        sum_fitness = sum(i[2] for i in cls.population)
        avg_fitness = sum_fitness / len(cls.population)
        print(f"세대 수 : {cls.generation} 최고 fitness : {cls.population[0][2]} 평균 fitness : {avg_fitness}")

    @classmethod
    def search(cls):
        population = [] # 해집단
        offsprings = [] # 자식해집단

        cls.init_population()
        cls.sort_population()
        while True:
            offsprings = []
            count_end = 0  # 동일 갯수
            for i in range(cls.NUM_OFFSPRING):


                mom_ch, dad_ch = cls.selection_operator()
                offspring1 = cls.crossover_operator_JCO(mom_ch, dad_ch)
                coin = random.random()
                if coin < cls.MUTATION_PROB:
                    offspring1 = cls.mutation_operator(offspring1)
                    offspring1 = cls.mutation_operator_bt_neck(offspring1)


                offsprings.append(offspring1)

            cls.replacement_operator(offsprings)
            cls.sort_population()
            cls.print_avg_fittness()
            cls.generation += 1

            if cls.generation == 10000:
                break

        print(cls.population[0])

#s = GA_allocate.fixed_seq()
#th = GA_allocate.get_fittness(s)
#print(th)
GA_allocate.search()