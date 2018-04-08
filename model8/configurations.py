import os


class Config(object):

    Run_Index = 'final_opt_round_2'

    # TODO: EXP
    Sample_binary_path = 'samples_binary_' + Run_Index
    Event_descriptions = 'event_desc.csv'

    # TODO: corpus / data / preprocessing
    Clean_corpus_folder = r'.\clean_data'
    Clean_corpus_6_folder = os.path.join('.', 'data6u')
    CORPUS_FOLDER = r'.\InScript_LREC2016\InScript\corpus'

    Checkpoint_path = 'ghost lol lol '
    Epsilon = 1e-12
    # thres for NORM clipping
    CSV_log_path = r'.\log' + Run_Index + '.csv'
    LOG_DIR = r'.\tbLOG'
    Sample_path = 'samples_' + Run_Index
    # for unix
    # LOG_DIR = r'./LOG'
    Word2Vec_Google_Path = r'D:\data\GoogleNews-vectors-negative300.bin'
    W2V_wrapper_path = os.path.join('.', 'w2v10')

    # Effective_Scripts = ['bath']
    # Effective_Scripts = ['bath', 'grocery', 'tree']
    Effective_Scripts = ['bath', 'bicycle', 'bus', 'cake', 'flight', 'grocery', 'haircut', 'library', 'train', 'tree']

    All_Scripts = ['bath', 'bicycle', 'bus', 'cake', 'flight', 'grocery', 'haircut', 'library', 'train', 'tree']
    Empty_event_label = 'empty'
    End_of_sentence_tokens = list(['.', '!'])
    End_of_story_label = '<end_of_story>'

    Evoking_Event = 'Evoking'
    Beginning_Event = 'Story_Begin'
    Ending_Event = 'Story_End'
    Invalid_Labels = ['1/2', '1/4']
    # Invalid_Labels = ['UnrelEv', 'RelNScrEv', 'Unclear', '1/2', '1/4', 'ScrEv_other']
    Post_fix_events = ['UnrelEv', 'RelNScrEv', 'Unclear', 'ScrEv_other', 'Evoking']
    # Post_fix_events = ['Evoking']
    Padding_token = '<pad>'
    Punktuations = [',', '.', '!', ':', ';', '?']
    # Punktuation_unified = '.'
    Concatenation_delimiter = '+'
    Universal_number = 'several'

    # for unix:
    # CORPUS_FOLDER = r'./InScript_LREC2016/InScript/corpus'

    # TODO: TEST STABILITY
    Validation_proportion = 0.025

    # TODO: model hypers----------------------------------------------------------------------
    # A_hidden_size = 512
    C_gru_interpolation_coef = 0.1
    # dropout for non-recurrent recurrent connections in RNN and dense layers
    Dropout_rate = 0.5
    # event embedding size equals to size of output o_t
    Event_embedding_size = 256
    Input_Embedding_size = 300

    Max_vocabulary_size = 99999

    # tokens strictly less frequent will be replaced by '<unk>'
    Least_frequency = 1
    # calling dataloader sets these values
    Active_vocabulary_size = 0
    # includes 'story begin', 'story end' labels. these are not included in the corpus.
    Event_vocabulary_size = 0

    RNN_size = 768
    Context_length = 20

    # training
    Batch_size = 256
    Learning_rate = 0.397e-3

    Clipping_threshold = 1e10
    Early_stopping_patience = 1
    Loss_weight_on_a = 1.3
    Max_train_epochs = 100
    # weights_on_cat_of_a_loss
    # wa =
    Weight_on_a1_cross_entropy = [1., 4]
    # TODO: end model hypers-------------------------------------------------------------------
    # TODO: Generation
    # lower temperature -> conservative
    Temperature = .5
    Crucial_temperature = .01
    # the least number of tokens to generate before shifting events. should go well with **C_gru only** models.
    Minimal_event_span = 1

    class Seed(object):
        def __init__(self, x_context, e_p_context, e_f_context, agenda):
            self.scripts = ['bath', 'bicycle', 'bus', 'cake', 'flight',
                            'grocery', 'haircut', 'library', 'train', 'tree']
            self.x_context = x_context
            self.e_p_context = e_p_context
            self.e_f_context = e_f_context
            self.agenda = agenda
            self.script = [s for s in self.scripts if self.agenda[0].find(s) != -1][0]

    x_context = 'after playing football with my friends , ' \
                'i was all sweaty and dirty , ' \
                'so i decided i should take a bath .'.split()
    e_p_context = ['Story_Begin_bath'] * 22 + ['Evoking_bath']
    e_f_context = ['Evoking_bath'] * 22 + ['ScrEv_wash']
    # agenda includes the events to be instantiated. It begins with the last event that was instantiated
    # in the seed.
    agenda_bath = list(['Evoking_bath', 'ScrEv_close_drain', 'ScrEv_turn_water_on',
                        'ScrEv_check_temp', 'ScrEv_undress', 'ScrEv_sink_water', 'ScrEv_relax',
                        'ScrEv_wash', 'ScrEv_get_out_bath', 'ScrEv_dry', 'Story_End_bath'])

    agenda_bathu = list(['Evoking_bath', 'UnrelEv_bath', 'ScrEv_close_drain', 'UnrelEv_bath',
                        'ScrEv_turn_water_on', 'UnrelEv_bath',
                        'ScrEv_check_temp', 'UnrelEv_bath',
                        'ScrEv_undress', 'UnrelEv_bath',
                        'ScrEv_sink_water', 'UnrelEv_bath',
                        'ScrEv_relax', 'UnrelEv_bath',
                        'ScrEv_wash', 'UnrelEv_bath',
                        'ScrEv_get_out_bath', 'UnrelEv_bath',
                        'ScrEv_dry', 'UnrelEv_bath',
                        'Story_End_bath'])

    Sample_Seed_bath = Seed(x_context, e_p_context, e_f_context, agenda_bath)

    x_context2 = 'yesterday i went+grocery+shopping .'.split()
    e_p_context2 = ['Story_Begin_grocery'] * 3 + ['Evoking_grocery']
    e_f_context2 = ['Evoking_grocery'] * 3 + ['ScrEv_take_bags']
    agenda_grocery = list(['Evoking_grocery', 'ScrEv_take_bags', 'ScrEv_take_shop_cart', 'ScrEv_enter',
                           'ScrEv_check_list', 'ScrEv_move_section', 'ScrEv_get_groceries', 'ScrEv_get_groceries',
                           'ScrEv_check_off', 'ScrEv_go_checkout', 'ScrEv_wait', 'ScrEv_put_conveyor',
                           'ScrEv_cashier_scan/weight', 'ScrEv_pack_groceries', 'ScrEv_pay', 'ScrEv_get_receipt',
                           'ScrEv_bring_vehicle', 'ScrEv_leave', 'Story_End_grocery'])
    Sample_Seed_grocery = Seed(x_context2, e_p_context2, e_f_context2, agenda_grocery)

    x_contextb2 = 'after playing football with my friends , '.split()
    e_p_contextb2 = ['Story_Begin_bath'] * 7
    e_f_contextb2 = ['Evoking_bath'] * 7
    # agenda includes the events to be instantiated. It begins with the last event that was instantiated
    # in the seed.
    agenda_bath = list(['Story_Begin_bath', 'Evoking_bath', 'ScrEv_close_drain', 'ScrEv_turn_water_on',
                        'ScrEv_check_temp', 'ScrEv_undress', 'ScrEv_sink_water', 'ScrEv_relax',
                        'ScrEv_wash', 'ScrEv_get_out_bath', 'ScrEv_dry', 'Story_End_bath'])

    Sample_Seed_bath2 = Seed(x_contextb2, e_p_contextb2, e_f_contextb2, agenda_bath)

    x_contextbb = 'after playing football with my friends , '.split()
    e_p_contextbb = ['Story_Begin_bath'] * 7
    e_f_contextbb = ['Evoking_bath'] * 7
    # agenda includes the events to be instantiated. It begins with the last event that was instantiated
    # in the seed.
    agenda_bathbb = list(['Story_Begin_bath', 'Evoking_bath', 'ScrEv_get_towel', 'ScrEv_take_clean_clothes',
                          'ScrEv_turn_water_on', 'ScrEv_turn_water_off',
                          'ScrEv_undress', 'ScrEv_sink_water', 'ScrEv_apply_soap',
                          'ScrEv_wash', 'ScrEv_get_out_bath', 'ScrEv_dry', 'ScrEv_leave', 'Story_End_bath'])

    x_contextgg = 'yesterday i went+grocery+shopping .'.split()
    e_p_contextgg = ['Story_Begin_grocery'] * 3 + ['Evoking_grocery']
    e_f_contextgg = ['Evoking_grocery'] * 3 + ['ScrEv_take_bags']
    agenda_grocerygg = list(['Evoking_grocery', 'ScrEv_take_bags', 'ScrEv_enter',
                           'ScrEv_check_list', 'ScrEv_get_groceries',
                           'ScrEv_check_off', 'ScrEv_go_checkout', 'ScrEv_wait', 'ScrEv_put_conveyor',
                           'ScrEv_cashier_scan/weight', 'ScrEv_pack_groceries', 'ScrEv_pay', 'ScrEv_get_receipt',
                           'ScrEv_bring_vehicle', 'ScrEv_return_shop_cart', 'ScrEv_leave', 'Story_End_grocery'])

    Sample_Seed_grocerygg = Seed(x_contextgg, e_p_contextgg, e_f_contextgg, agenda_grocerygg)

    Sample_Seed_bathb = Seed(x_contextbb, e_p_contextbb, e_f_contextbb, agenda_bathbb)

    Sample_seed_bathu = Seed(x_contextb2, e_p_contextb2, e_f_contextb2, agenda_bathu)

    Seeds = [Sample_Seed_bath2, Sample_Seed_grocery, Sample_Seed_bathb, Sample_Seed_grocerygg]
