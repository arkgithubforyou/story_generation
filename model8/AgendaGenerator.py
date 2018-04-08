import dill
import configurations
from configurations import Config
import numpy as np
import random

tree = [
    ['choose_tree'],
    ['get_tree'],
    ['get_tools'],
    ['take_home'],
    ['find_place'],
    ['dig_hole'],
    ['place_fertilizers', 'unwrap_root'],
    ['place_root'],
    ['refill_hole'],
    ['tamp_dirt'],
    ['water'],
]
bicycle = [
    ['lay_bike_down', 'get_tools', 'loose_nut', 'pull_air_pin'],
    ['remove_tire'],
    ['get_tire', 'take_tire_off'],
    ['examine_tire'],
    ['put_patch/seal'],
    ['put_new_tire'],
    ['refill_tire_air'],
    ['check_new_tire'],
    ['ride_bike']
]
bus = [
    ['check_time-table', 'find_bus'],
    ['get_bus_stop'],
    ['wait'],
    ['bus_comes'],
    ['board_bus'],
    ['get_ticket'],
    ['find_place'],
    ['ride', 'spend_time_bus'],
    ['press_stop_button'],
    ['bus_stops'],
    ['go_exit'],
    ['get_off'],
]
bath = [
    ['enter_bathroom', 'get_towel'],
    ['prepare_bath'],
    ['take_clean_clothes', 'turn_water_on'],
    ['check_temp', 'close_drain'],
    ['put_bubble_bath_scent', 'fill_water/wait'],
    ['turn_water_off', 'undress'],
    ['sink_water'],
    ['apply_soap', 'relax'],
    ['wash'],
    ['get_out_bath', 'open_drain'],
    ['dry'],
    ['get_dressed', 'leave'],
]
grocery = [
    ['make_list'],
    ['go_grocery'],
    ['enter'],
    ['take_shop_cart'],
    ['move_section'],
    ['get_groceries', 'check_off'],
    ['check_list', 'go_checkout'],
    ['cashier_scan/weight', 'wait'],
    ['put_conveyor'],
    ['pay'],
    ['pack_groceries', 'get_receipt', 'bring_vehicle'],
    ['return_shop_cart'],
    ['leave']
]
haircut = [
    ['make_appointment'],
    ['get_salon'],
    ['enter'],
    ['check_in'],
    ['wait'],
    ['sit_down'],
    ['put_on_cape'],
    ['talk_haircut'],
    ['move_in_salon', 'wash'],
    ['comb'],
    ['cut'],
    ['brush_hair', 'dry', 'make_hair_style'],
    ['look_mirror'],
    ['customer_opinion'],
    ['pay'],
    ['leave_tip'],
    ['leave'],
]
flight = [
    ['get_ticket', 'present_ID/ticket'],
    ['pack_luggage'],
    ['get_airport'],
    ['go_check_in'],
    ['check_in'],
    ['check_luggage', 'go_security_checks'],
    ['find_terminal'],
    ['wait_boarding'],
    ['present_boarding_pass'],
    ['board_plane'],
    ['stow_away_luggage', 'take_seat'],
    ['buckle_seat_belt', 'listen_crew'],
    ['take_off_preparations'],
    ['take_off'],
    ['spend_time_flight'],
    ['landing'],
    ['exit_plane'],
    ['retrieve_luggage']
]
library = [
    ['get_library', 'browse_releases'],
    ['ask_librarian', 'get_shelf'],
    ['look_for_book', 'obtain_card'],
    ['use_computer'],
    ['check_catalog'],
    ['note_shelf', 'take_book'],
    ['go_check_out'],
    ['show_card'],
    ['register'],
    ['get_receipt'],
    ['leave'],
    ['return_book'],
]
cake = [
    ['choose_recipe'],
    ['get_ingredients'],
    ['preheat'],
    ['get_utensils'],
    ['add_ingredients'],
    ['prepare_ingredients'],
    ['grease_cake_tin'],
    ['pour_dough'],
    ['put_cake_oven'],
    ['set_time'],
    ['wait'],
    ['check', 'take_out_oven'],
    ['turn_off_oven', 'cool_down', 'take_out_cake_tin'],
    ['decorate'],
    ['eat'],
]
train = [
    ['check_time-table'],
    ['get_train_station'],
    ['get_tickets'],
    ['get_platform'],
    ['wait'],
    ['train_arrives'],
    ['get_on'],
    ['find_place', 'conductor_checks', 'arrive_destination'],
    ['spend_time_train'],
    ['get_off'],
]


class Agenda(object):
    script_representations = {'bath': bath, 'bicycle': bicycle,
                              'bus': bus, 'cake': cake, 'flight': flight,
                              'grocery': grocery, 'haircut': haircut, 'library': library,
                              'train': train, 'tree': tree}
    event_counter = dill.load(open('e_counters', 'rb'))

    @staticmethod
    def generate_agenda(script, temperature=0.5):
        dfa = Agenda.script_representations[script]
        count = Agenda.event_counter[script]
        agenda = list()
        agenda.append('Story_Begin_' + script)
        agenda.append('Evoking_' + script)
        for segment in dfa:
            for event in segment:
                sample = np.random.uniform(0, 1)
                if sample < np.power(min(count[event]/100, 1), temperature):
                    agenda.append('ScrEv_' + event)
        agenda.append('Story_End_' + script)
        return agenda

    @staticmethod
    def generate_random_agenda(script, length):
        dfa = Agenda.script_representations[script]
        agenda = list()
        agenda.append('Story_Begin_' + script)
        agenda.append('Evoking_' + script)
        for _ in range(length):
            candidate_list = random.choice(dfa)
            random_event = random.choice(candidate_list)
            agenda.append('ScrEv_' + random_event)
        agenda.append('Story_End_' + script)
        return agenda

    @staticmethod
    def generate_seed(script, temperature=1):
        # def __init__(self, x_context, e_p_context, e_f_context, agenda)
        x_context = list(['yesterday', 'i'])
        e_p_context = ['Story_Begin_' + script] * 2
        e_f_context = ['Evoking_' + script] * 2
        agenda = Agenda.generate_agenda(script, temperature)
        return Config.Seed(x_context, e_p_context, e_f_context, agenda)

    @staticmethod
    def generate_random_seed(script, length):
        x_context = list(['yesterday', 'i'])
        e_p_context = ['Story_Begin_' + script] * 2
        e_f_context = ['Evoking_' + script] * 2
        agenda = Agenda.generate_random_agenda(script, length)
        return Config.Seed(x_context, e_p_context, e_f_context, agenda)

    @staticmethod
    def generate_seeds(scripts, lengths):
        """
        generate 4 seeds for each script: 2 rational and 2 random of length lengths[script]
        :param scripts:
        :param lengths: dict script->length of agenda
        :return:
        """
        seeds = list()
        for script in scripts:
            seeds.append(Agenda.generate_random_seed(script, lengths[script]))
            seeds.append(Agenda.generate_random_seed(script, lengths[script]))
            seeds.append(Agenda.generate_seed(script))
            seeds.append(Agenda.generate_seed(script))
        return seeds


ss = Agenda.generate_seeds(['grocery'], {'grocery': 15})

# aa = Agenda.generate_random_seed('bath', 10)

'''
for script in Agenda.script_representations:
    for segment in Agenda.script_representations[script]:
        for event in segment:
            ee = Agenda.event_counter[script]
            if event not in ee:
                print('wo ha ha')
'''


pass
