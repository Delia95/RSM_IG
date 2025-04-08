# # pip install scikit-opt
import sko.GA as GA
import pandas as pd

class Car:
    def __init__(self, id, power, driver):
        self.ID = id
        # 1代表燃油；2代表混动
        self.power = power
        # 1代表两驱；2代表四驱
        self.driver = driver
        # action
        self.action_list = []

class Transverse_moving:
    def __init__(self):
        self.state = -1
        self.cost = 0
        self.car_id = 0


class Lane:
    def __init__(self, id, cost, return_cost):
        self.ID = id
        self.parking_spot = [0] * 10
        self.parking_spot_state_time = [-1] * 10
        self.cost = cost
        self.return_cost = return_cost


def get_reward(solutions):
    for solution in solutions:
        solution_list = []
        while solution != 0:
            temp = solution % 6
            solution = solution // 6
            solution_list.append(temp)
        print(solution_list)


class Environment:
    def __init__(self):
        self.car_list = []
        self.lane_list = []
        self.cost_list = [18, 12, 6, 0, 12, 18]
        self.return_cost_list = [24, 18, 12, 6, 12, 18]
        self.time = 0
        # 接车横移机
        self.transverse_moving1 = Transverse_moving()
        # 送车横移机
        self.transverse_moving2 = Transverse_moving()
        # 返回道
        self.return_lane = Lane(7, self.return_cost_list, 0)

    def read_cars(self):
        #D2.xlsx
        data = pd.read_excel('D1.xlsx')
        for i in range(len(data)):
            id = int(data["进车顺序"][i])
            if data["动力"][i] == "燃油":
                power = 1
            else:
                power = 2
            if data["驱动"][i] == "两驱":
                driver = 1
            else:
                driver = 2
            self.car_list.append(Car(id, power, driver))

    def define_lanes(self):
        for i in range(6):
            self.lane_list.append(Lane(i, self.cost_list[i], self.return_cost_list[i]))

    #
    def moving_one_step(self, car, car_out_list, car_out_order, waiting_flag):
        '''
        让系统运行1秒
        :return:
        '''
        # 用以标注是否要更换car
        flag = 1
        # 模拟车辆在lane上随时间的移动
        for lane in self.lane_list:
            for i in range(1, 10):
                if lane.parking_spot[i] == 0:
                    continue
                else:
                    if lane.parking_spot_state_time[i] != 0:
                        lane.parking_spot_state_time[i] = lane.parking_spot_state_time[i] - 1
                    if lane.parking_spot_state_time[i] == 0 and lane.parking_spot[i - 1] == 0:
                        lane.parking_spot[i - 1] = lane.parking_spot[i]
                        lane.parking_spot[i] = 0
                        if i != 1:
                            lane.parking_spot_state_time[i - 1] = 9
                        else:
                            lane.parking_spot_state_time[i - 1] = 0
            if lane.parking_spot[0] != 0:
                lane.parking_spot_state_time[0] = lane.parking_spot_state_time[0] - 1
        # 处理返回道上的车辆
        for i in range(8, -1, -1):
            if self.return_lane.parking_spot[i] == 0:
                continue
            else:
                if self.return_lane.parking_spot_state_time[i] != 0:
                    self.return_lane.parking_spot_state_time[i] = self.return_lane.parking_spot_state_time[i] - 1
                if self.return_lane.parking_spot_state_time[i] == 0 and self.return_lane.parking_spot[i + 1] == 0:
                    self.return_lane.parking_spot[i + 1] = self.return_lane.parking_spot[i]
                    self.return_lane.parking_spot[i] = 0
                    if i != 8:
                        self.return_lane.parking_spot_state_time[i + 1] = 9
                    else:
                        self.return_lane.parking_spot_state_time[i + 1] = 0

        # transverse_moving1
        if self.transverse_moving1.state == -1:
            if self.return_lane.parking_spot[9] == 0:
                if car is not None:
                    if waiting_flag == 0:
                        self.transverse_moving1.state = car.action_list[0]
                        car_action=car.action_list[0]
                        if car_action>len(self.cost_list):
                            car_action=0
                        self.transverse_moving1.cost = self.cost_list[car_action]
                        self.transverse_moving1.car_id = car.ID
                        del car.action_list[0]
                    # 说明汽车已经被放到横移机上，可以考虑下一辆汽车了
                        flag = 0
            else:
                this_car = self.car_list[self.return_lane.parking_spot[9] - 1]
                self.transverse_moving1.state = this_car.action_list[0]
                self.transverse_moving1.cost = self.return_cost_list[this_car.action_list[0]]
                self.transverse_moving1.car_id = this_car.ID
                self.return_lane.parking_spot[9] = 0
                del this_car.action_list[0]
        else:
            self.transverse_moving1.cost = self.transverse_moving1.cost - 1
            if self.transverse_moving1.cost <= 0:
                # 尝试将汽车放到lane上
                moving_state = self.transverse_moving1.state
                if moving_state > len(self.lane_list):
                    moving_state = 0
                if self.lane_list[moving_state].parking_spot[9] != 0:
                    self.transverse_moving1.cost = 0
                else:
                    self.lane_list[moving_state].parking_spot[9] = self.transverse_moving1.car_id
                    self.lane_list[moving_state].parking_spot_state_time[9] = 9
                    self.moving_state = -1
                    this_car = self.car_list[self.transverse_moving1.car_id - 1]
                    if len(this_car.action_list) <= 0:
                        car_out_order.append(this_car)


        # transverse_moving2
        if self.transverse_moving2.state == -1:
            min_lane_id = -1
            min_cost = 1
            for i in range(len(self.lane_list)):
                if self.lane_list[i].parking_spot[0] != 0:
                    if self.lane_list[i].parking_spot_state_time[0] < min_cost:
                        min_cost = self.lane_list[i].parking_spot_state_time[0]
                        min_lane_id = i
            if min_lane_id != -1:
                self.transverse_moving2.state = min_lane_id
                if len(self.car_list[self.lane_list[min_lane_id].parking_spot[0] - 1].action_list) == 0:
                    self.transverse_moving2.cost = self.cost_list[min_lane_id]
                    self.transverse_moving2.car_id = self.lane_list[min_lane_id].parking_spot[0]
                    self.lane_list[min_lane_id].parking_spot[0] = 0
                    self.lane_list[min_lane_id].parking_spot_state_time[0] = 0
                else:
                    self.transverse_moving2.cost = self.return_cost_list[min_lane_id]
                    self.transverse_moving2.car_id = self.lane_list[min_lane_id].parking_spot[0]
                    self.lane_list[min_lane_id].parking_spot[0] = 0
                    self.lane_list[min_lane_id].parking_spot_state_time[0] = 0
        else:
            self.transverse_moving2.cost = self.transverse_moving2.cost - 1
            if self.transverse_moving2.cost <= 0:
                # 将汽车从送车横移机上卸下送到总装车间或者进入返回道
                if len(self.car_list[self.transverse_moving2.car_id - 1].action_list) == 0:
                    # 进入总装车间
                    car_out_list.append(self.transverse_moving2.car_id)
                    self.transverse_moving2.state = -1
                else:
                    if self.return_lane.parking_spot[0] == 0:
                        self.return_lane.parking_spot[0] = self.transverse_moving2.car_id
                        self.return_lane.parking_spot_state_time[0] = 9
                        self.transverse_moving2.state = -1
        if car is None:
            for lane in self.lane_list:
                for spot in lane.parking_spot:
                    if spot != 0:
                        return 1, flag, car_out_list, car_out_order
            for spot in self.return_lane.parking_spot:
                if spot != 0:
                    return 1, flag, car_out_list, car_out_order
            if self.transverse_moving2.state != -1:
                return 1, flag, car_out_list, car_out_order
            if self.transverse_moving1.state != -1:
                return 1, flag, car_out_list, car_out_order
            return -1, flag, car_out_list, car_out_order
        else:
            return 1, flag, car_out_list, car_out_order


    def check_score1_score2(self, car_order_list):
        temp_score1 = 100
        temp_score2 = 100
        last = -1
        now = -1
        if self.car_list[car_order_list[0] - 1].power != 2:
            temp_score1 = temp_score1 - 1
        k = 0
        for i in range(len(car_order_list)):
            if self.car_list[car_order_list[i] - 1].power == 2:
                if last == -1:
                    last = k
                else:
                    now = k
                    if now - last != 2:
                        temp_score1 = temp_score1 - 1
                    last = now
            k = k + 1

        # 计算优化目标2
        flag_driver = self.car_list[car_order_list[0] - 1].driver
        flag_driver_change = 0
        count1 = 0
        count2 = 0
        k = 0
        for i in range(len(car_order_list)):
            if self.car_list[car_order_list[i] - 1].driver == flag_driver and flag_driver_change == 0:
                count1 = count1 + 1
                continue
            if self.car_list[car_order_list[i] - 1].driver != flag_driver and flag_driver_change == 0:
                flag_driver_change = 1
                count2 = count2 + 1
                continue
            if self.car_list[car_order_list[i] - 1].driver == flag_driver and flag_driver_change == 1:
                # 计算扣分还是得分
                if count1 != count2:
                    temp_score2 = temp_score2 - 1
                count1 = 0
                count2 = 0
                flag_driver_change = 0
                i = i - 1
        return 0.4 * temp_score1 + 0.3 * temp_score2


    def get_reward(self, solution):
        car_out_order = []
        solution = solution.astype('int')
        score3 = 100

        car_num = len(self.car_list)
        # 将解决方案导入到car的动作中
        for i in range(car_num):
            this_solution = solution[i]
            if this_solution <= 5:
                self.car_list[i].action_list = [this_solution]
            else:
                this_1 = this_solution - 5
                temp_action_list = []

                temp_action_list.append(this_1)
                temp_action_list.append(3)
                self.car_list[i].action_list = temp_action_list
                score3 = score3 - 1

        car_out_list = []
        time = 0
        car_i = 0
        waiting_flag = 0
        while True:
            time = time + 1
            if car_i >= car_num:
                car = None
            else:
                car = self.car_list[car_i]
            if len(car_out_order) > 0 and sum(self.return_lane.parking_spot) > 0:
                first_car = None
                first_j = 0
                for j in range(8, -1, -1):
                    if self.return_lane.parking_spot[j] > 0:
                        first_car = self.car_list[self.return_lane.parking_spot[j] - 1].ID
                        first_j = j
                        break
                temp_list3 = []
                temp_list1 = []
                temp_list2 = []
                for j in range(len(car_out_order)):
                    temp_list1.append(car_out_order[j].ID)
                    temp_list2.append(car_out_order[j].ID)
                    temp_list3.append(car_out_order[j].ID)
                if first_car is not None:
                    temp_list1.append(first_car)
                    if car is not None:
                        temp_list2.append(car.ID)
                    temp_1 = self.check_score1_score2(temp_list1) - 0.01 * 0.1 * (9 * (8 - first_j) + self.return_lane.parking_spot_state_time[first_j] + 6 * 2)
                    temp_2 = self.check_score1_score2(temp_list2)
                    if car is not None:
                        temp_3 = self.check_score1_score2(temp_list3) - 0.01 * 0.1 * (6 + self.return_cost_list[car.action_list[0]]) - 0.2
                    else:
                        temp_3 = temp_2 - 0.2
                    if temp_1 > temp_2 and temp_1 > temp_3:
                        waiting_flag = 1
                    if temp_2 > temp_1 and temp_2 > temp_3:
                        waiting_flag = 0
                    if temp_3 > temp_1 and temp_3 - 0.1 > temp_2 and score3 > 80:
                        waiting_flag = 0
                        if len(car.action_list) < 2:
                            car.action_list.append(3)
                            score3 = score3 - 1
            else:
                if len(car_out_order) > 0:
                    temp_list3 = []
                    temp_list2 = []
                    for j in range(len(car_out_order)):
                        temp_list2.append(car_out_order[j].ID)
                        temp_list3.append(car_out_order[j].ID)
                    if car is not None:
                        temp_list2.append(car.ID)
                        temp_2 = self.check_score1_score2(temp_list2)
                        temp_3 = self.check_score1_score2(temp_list3) - 0.01 * 0.1 * (6 + self.return_cost_list[car.action_list[0]]) - 0.2
                        if temp_3 - 0.1 > temp_2 and score3 > 80:
                            if len(car.action_list) < 2:
                                car.action_list.append(3)
                                score3 = score3 - 1
                waiting_flag = 0
            flag1, flag, car_out_list, car_out_order = self.moving_one_step(car, car_out_list, car_out_order, waiting_flag)
            if flag1 == -1:
                break
            if flag == 0:
                car_i = car_i + 1
        # 根据目标函数计算得分
        # 计算优化目标1
        score1 = 100
        last = -1
        now = -1
        if self.car_list[car_out_list[0] - 1].power != 2:
            score1 = score1 - 1
        k = 0
        for i in range(len(car_out_list)):
            if self.car_list[car_out_list[i] - 1].power == 2:
                if last == -1:
                    last = k
                else:
                    now = k
                    if now - last != 2:
                        score1 = score1 - 1
                    last = now
            k = k + 1
        # 计算优化目标2
        score2 = 100
        flag_driver = self.car_list[car_out_list[0] - 1].driver
        flag_driver_change = 0
        count1 = 0
        count2 = 0
        k = 0
        for i in range(len(car_out_list)):
            if self.car_list[car_out_list[i] - 1].driver == flag_driver and flag_driver_change == 0:
                count1 = count1 + 1
                continue
            if self.car_list[car_out_list[i] - 1].driver != flag_driver and flag_driver_change == 0:
                flag_driver_change = 1
                count2 = count2 + 1
                continue
            if self.car_list[car_out_list[i] - 1].driver == flag_driver and flag_driver_change == 1:
                # 计算扣分还是得分
                if count1 != count2:
                    score2 = score2 - 1
                count1 = 0
                count2 = 0
                flag_driver_change = 0
                i = i - 1
        # 计算优化目标4
        score4 = 100 - 0.01 * (time - 9 * car_num - 72)
        score = 0.4 * score1 + 0.3 * score2 + 0.2 * score3 + 0.1 * score4
        print(score, score1*0.4, score2*0.3, score3*0.2, score4*0.1, time, len(car_out_list), len(car_out_order), flag1)
        return -score

if __name__ == '__main__':
    env = Environment()
    env.read_cars()
    env.define_lanes()
    car_num = len(env.car_list)
    lb = []
    ub = []
    stack_list = []
    for i in range(car_num):
        lb.append(0)
        ub.append(555)
    # 采用遗传算法选择最佳动作值
    ga = GA.GA(func=env.get_reward, n_dim=len(lb), size_pop=100, max_iter=500, lb=lb, ub=ub, prob_mut=0.5,
                precision=1)
    action, best_reward = ga.run()

    print("best:")
    print(best_reward)