from configurations import Config

import keras
import dataLoaders
import collections
import utils
import numpy as np
import os
import tensorflow as tf
import dill
import copy

import arksKerasTools
import AgendaGenerator


# Global Variables: data and dicts
x_train, y_train, e_p_train, e_f_train, e_p_context_train, e_f_context_train, a_train, \
    x_val, y_val, e_p_val, e_f_val, e_p_context_val, e_f_context_val, a_val, \
    di2w, dw2i, di2e, de2i, _ = dataLoaders.model_6_loader()
w2v_matrix = dataLoaders.load_word2vec_matrix()
c_gru_coef_train = Config.C_gru_interpolation_coef * np.ones(shape=[np.shape(x_train)[0], 1])
c_gru_coef_val = Config.C_gru_interpolation_coef * np.ones(shape=[np.shape(x_val)[0], 1])

bath_random_seed = AgendaGenerator.Agenda.generate_seed('bath', temperature=1)
grocery_random_seed = AgendaGenerator.Agenda.generate_seed('grocery', temperature=1)
Config.Seeds.extend([bath_random_seed, grocery_random_seed])


def generate_sequences(model, seed, generation_length, n_generation=1,
                       temperature=Config.Temperature, context_length=Config.Context_length,
                       minimal_span=Config.Minimal_event_span):
    """
    generates sequence with given seed in TEXT.
    :param
    model: a trained model
    seed: a configuration.Seed object.
    :return: list of generated sequences together with the seed, converted to text.
    """
    generated_sequence_s = list()
    a_list = list()
    e_p_histories = list()
    e_f_histories = list()
    text_lists = list()
    for i in range(n_generation):
        # initialize stuff
        e_p_pointer = 0
        e_f_pointer = 1
        x_context = collections.deque(seed.x_context, maxlen=context_length)
        e_p_context = collections.deque(seed.e_p_context, maxlen=context_length)
        e_f_context = collections.deque(seed.e_f_context, maxlen=context_length)
        e_p_history = list()
        e_f_history = list()

        generated_sequence = seed.x_context[:]
        a_outs = list()

        event_span = 1

        for j in range(generation_length):
            x_seq_input_int = dataLoaders.word_to_int(x_context, dw2i)
            x_seq_input = np.reshape(np.array(x_seq_input_int), newshape=(1, -1))
            e_p_context_input_int = dataLoaders.word_to_int(e_p_context, de2i)
            e_p_context_input = np.reshape(e_p_context_input_int, newshape=(1, -1))
            e_f_context_input_int = dataLoaders.word_to_int(e_f_context, de2i)
            e_f_context_input = np.reshape(e_f_context_input_int, newshape=(1, -1))
            e_f_input = np.array([de2i[seed.agenda[e_f_pointer]]])
            e_p_input = np.array([de2i[seed.agenda[e_p_pointer]]])

            c_coef_temp = np.array([Config.C_gru_interpolation_coef])

            a_temp = np.array([[.5, .5]])

            # run the model to get a
            _, a_out = model.predict([x_seq_input, e_p_context_input,
                                      e_f_context_input, e_f_input, e_p_input, a_temp, c_coef_temp])
            # with the predicted a, run the model again to get output
            pred_distribution, a_out = model.predict([x_seq_input, e_p_context_input,
                                                      e_f_context_input, e_f_input, e_p_input, a_out, c_coef_temp])

            if a_out[0][1] > a_out[0][0] and event_span >= minimal_span:
                # sample event word
                index_of_new_word = utils.temperature_sample(pred_distribution, temperature=Config.Crucial_temperature)
            else:
                # sample normal word
                index_of_new_word = utils.temperature_sample(pred_distribution, temperature=temperature)

            new_word = di2w[index_of_new_word]

            a_outs.append(a_out)
            x_context.append(new_word)
            e_p_context.append(seed.agenda[e_p_pointer])
            e_f_context.append(seed.agenda[e_f_pointer])
            e_p_history.append(seed.agenda[e_p_pointer])
            e_f_history.append(seed.agenda[e_f_pointer])

            generated_sequence.append(new_word)

            # shift to the next event if span reached and generation is not finishing
            if a_out[0, 1] > a_out[0, 0] and event_span >= minimal_span and\
                    seed.agenda[e_f_pointer].find(Config.Ending_Event) == -1:
                e_p_pointer += 1
                e_f_pointer += 1
                event_span = 0

            event_span += 1

            # keep generating until all events expanded and a punktuation is met.
            if seed.agenda[e_f_pointer].find(Config.Ending_Event) != -1 and new_word in Config.Punktuations:
                break
            # end for per item

        # filter generated sequence to remove repititions
        # MAX_REPITITION_SPAN = 3
        index = 0
        text_list = list()
        while index < len(generated_sequence):
            chunk_size = 1
            if index < len(generated_sequence) - 1:
                if generated_sequence[index] == generated_sequence[index + 1]:
                    index += 1
            elif index < len(generated_sequence) - 3:
                if ' '.join(generated_sequence[index: index + 2]) == ' '.join(generated_sequence[index + 2: index + 4]):
                    index += 2
                    chunk_size = 2
            elif index < len(generated_sequence) - 5:
                if ' '.join(generated_sequence[index: index + 3]) == ' '.join(generated_sequence[index + 3: index + 6]):
                    index += 3
                    chunk_size = 3
            for ii in range(chunk_size):
                text_list.append(generated_sequence[index + ii])
            index += chunk_size

        generated_sequence_s.append(generated_sequence)
        a_list.append(a_outs)
        e_p_histories.append(e_p_history)
        e_f_histories.append(e_f_history)
        # end for per seq

        # filter out the '+' to generate text.
        text_lists.append(' '.join(text_list).replace('+', ' '))
    return generated_sequence_s, a_list, e_p_histories, e_f_histories, len(seed.x_context), text_lists


def plain_beam(model, seed, context_length=Config.Context_length, k=3):
    """
    plain beam search.
    :return text, seed pair
    """
    x_context = collections.deque(seed.x_context, maxlen=context_length)
    e_p_context = collections.deque(seed.e_p_context, maxlen=context_length)
    e_f_context = collections.deque(seed.e_f_context, maxlen=context_length)

    # initialize agenda with beginning configs
    beam_agenda = list()

    e_p_pointer = 0
    e_f_pointer = 1
    x_seq_input_int = dataLoaders.word_to_int(x_context, dw2i)
    # x_seq_input = np.reshape(np.array(x_seq_input_int), newshape=(1, -1))
    e_p_context_input_int = dataLoaders.word_to_int(e_p_context, de2i)
    # e_p_context_input = np.reshape(e_p_context_input_int, newshape=(1, -1))
    e_f_context_input_int = dataLoaders.word_to_int(e_f_context, de2i)
    # e_f_context_input = np.reshape(e_f_context_input_int, newshape=(1, -1))
    e_f_input = np.array([de2i[seed.agenda[e_f_pointer]]])
    e_p_input = np.array([de2i[seed.agenda[e_p_pointer]]])

    c_coef_temp = np.array([Config.C_gru_interpolation_coef])

    a_temp = np.array([[.5, .5]])

    # 'text' is a list
    agenda_item = {'x': x_seq_input_int, 'epc': e_p_context_input_int, 'efc': e_f_context_input_int, 'epp': e_p_pointer,
                   'efp': e_f_pointer, 'ep': e_p_input, 'ef': e_f_input, 'a': a_temp, 'c': c_coef_temp,
                   'text': seed.x_context, 'log_prob': 0, 'terminate': False}

    beam_agenda.append(agenda_item)

    while True:
        # beam search.
        # in each step, cache the candidates in cache_agenda and select the top ones to replace beam_agenda.
        # stops if seed.agenda is exhausted and a '.' is seen.

        cache_agenda = list()

        for aitem in beam_agenda:
            # run the model to get a
            xnp = np.reshape(np.array(aitem['x']), newshape=(1, -1))
            epcnp = np.reshape(aitem['epc'], newshape=(1, -1))
            efcnp = np.reshape(aitem['efc'], newshape=(1, -1))

            _, a_out = model.predict([xnp, epcnp, efcnp, aitem['ef'], aitem['ep'],
                                      aitem['a'], aitem['c']])
            # with the predicted a, run the model again to get output
            pred_distribution, a_out = model.predict([xnp, epcnp, efcnp, aitem['ef'], aitem['ep'], a_out, aitem['c']])

            top_k_indices = np.argsort(pred_distribution[0])[-k:][::-1]

            for index in top_k_indices:
                cache_item = copy.deepcopy(aitem)

                new_word = di2w[index]
                step_probability = pred_distribution[0][index]

                # update new item
                cache_item['x'].append(index)
                cache_item['epc'].append(de2i[seed.agenda[cache_item['epp']]])
                cache_item['efc'].append(de2i[seed.agenda[cache_item['efp']]])
                # shift to the next event if span reached and generation is not finishing
                if a_out[0, 1] > a_out[0, 0] and seed.agenda[cache_item['efp']].find(Config.Ending_Event) == -1:
                    if cache_item['efp'] >= len(seed.agenda) - 1:
                        cache_item['terminate'] = True
                    else:
                        cache_item['epp'] += 1
                        cache_item['efp'] += 1
                cache_item['ep'] = np.array([de2i[seed.agenda[cache_item['epp']]]])
                try:
                    cache_item['ef'] = np.array([de2i[seed.agenda[cache_item['efp']]]])
                except IndexError:
                    print('index error')
                cache_item['text'].append(new_word)
                cache_item['log_prob'] += np.log(step_probability)

                cache_agenda.append(cache_item)

        cache_agenda_sorted = sorted(cache_agenda, key=(lambda x: x['log_prob']))
        beam_agenda = cache_agenda_sorted[-k:]

        # terminate search if termination conditions are met
        break_flag = False
        for i in range(len(beam_agenda)):
            aitem = beam_agenda[i]
            if (aitem['text'][-1] in ['.', '!', '?'] and aitem['efp'] == len(seed.agenda) - 1) or\
                    (aitem['terminate'] is True):
                beam_agenda = [aitem]
                break_flag = True
                break
        if break_flag is True:
            break

    # filter generated sequence to remove repititions
    # MAX_REPITITION_SPAN = 3
    output_item = beam_agenda[0]
    text = output_item['text']

    index = 0
    filtered_text = list()
    while index < len(text):
        chunk_size = 1
        if index < len(text) - 1:
            if text[index] == text[index + 1]:
                index += 1
        elif index < len(text) - 3:
            if ' '.join(text[index: index + 2]) == ' '.join(text[index + 2: index + 4]):
                index += 2
                chunk_size = 2
        elif index < len(text) - 5:
            if ' '.join(text[index: index + 3]) == ' '.join(text[index + 3: index + 6]):
                index += 3
                chunk_size = 3
        for ii in range(chunk_size):
            filtered_text.append(text[index + ii])
        index += chunk_size

    # filter out the '+' in generated text.
    filtered_text = ' '.join(text).replace('+', ' ')
    return filtered_text, seed


def print_generation(seq, a_list, e_f_histories, seedlen, text_lists, path):
    fout = open(path, 'a')
    for j in range(len(seq)):
        fout.write('----SEQ:' + str(j) + ':\n')
        print('----SEQ:', j, ':')
        fout.write(('seed: ' + str(seq[0][:seedlen])) + '\n')
        print('seed: ', seq[0][:seedlen])
        for i in range(len(a_list[j])):
            fout.write(str(seq[j][i+seedlen]) + ': ' + str(a_list[j][i]) + ': ' + str(e_f_histories[j][i]) + '\n')
            print(str(seq[j][i+seedlen]) + ': ' + str(a_list[j][i]) + ': ' + str(e_f_histories[j][i]))
        print('----------------------------------------')
    for j in range(len(seq)):
        fout.write('----SEQ:' + str(j) + ':\n')
        fout.write(text_lists[j])
        fout.write('\n')
        fout.write(('seed: ' + str(seq[0][:seedlen])) + '\n')
    fout.close()


class SampleGeneration(keras.callbacks.Callback):
    def __init__(self, seed):
        self.seed = seed
        keras.callbacks.Callback.__init__(self)
        self.samples = list()

    def on_epoch_end(self, batch, logs=None):
        seq, a_list, e_p_histories, e_f_histories, seedlen, text_lists = \
            generate_sequences(self.model, seed=self.seed, generation_length=500, n_generation=1)
        self.samples.append(text_lists[0])
        seq, a_list, e_p_histories, e_f_histories, seedlen, text_lists = \
            generate_sequences(self.model, seed=Config.Sample_Seed_grocery, generation_length=500, n_generation=1)
        self.samples.append(text_lists[0])

        print()
        print(str(seq))
        print('seed=,', seq[0][:seedlen])
        for i in range(len(a_list[0])):
            print(str(seq[0][i+seedlen]) + ': ' + str(a_list[0][i]) + ': ' + str(e_f_histories[0][i]))


def r_0(y_true, y_pred):
    y_pred_0 = tf.round(y_pred)[:, 0]
    y_true_0 = y_true[:, 0]
    # p = tf.reduce_sum(tf.multiply(y_pred_0, y_true_0)) / (tf.reduce_sum(y_pred_0) + 0.000001)
    r = tf.reduce_sum(tf.multiply(y_pred_0, y_true_0)) / (tf.reduce_sum(y_true_0) + Config.Epsilon)
    return r


def f_0(y_true, y_pred):
    y_pred_0 = tf.round(y_pred)[:, 0]
    y_true_0 = y_true[:, 0]
    p = tf.reduce_sum(tf.multiply(y_pred_0, y_true_0)) / (tf.reduce_sum(y_pred_0) + Config.Epsilon)
    r = tf.reduce_sum(tf.multiply(y_pred_0, y_true_0)) / (tf.reduce_sum(y_true_0) + Config.Epsilon)
    return 2 * p * r / (p + r)


def r_1(y_true, y_pred):
    y_pred_1 = tf.round(y_pred[:, 1])
    y_true_1 = y_true[:, 1]
    # p = tf.reduce_sum(tf.multiply(y_pred_1, y_true_1)) / (tf.reduce_sum(y_pred_1) + 0.000001)
    r = tf.reduce_sum(tf.multiply(y_pred_1, y_true_1)) / (tf.reduce_sum(y_true_1) + Config.Epsilon)
    return r


def p_1(y_true, y_pred):
    y_pred_1 = tf.round(y_pred[:, 1])
    y_true_1 = y_true[:, 1]
    p = tf.reduce_sum(tf.multiply(y_pred_1, y_true_1)) / (tf.reduce_sum(y_pred_1) + Config.Epsilon)
    return p


def f_1(y_true, y_pred):
    y_pred_1 = tf.round(y_pred[:, 1])
    y_true_1 = y_true[:, 1]
    p = tf.reduce_sum(tf.multiply(y_pred_1, y_true_1)) / (tf.reduce_sum(y_pred_1) + Config.Epsilon)
    r = tf.reduce_sum(tf.multiply(y_pred_1, y_true_1)) / (tf.reduce_sum(y_true_1) + Config.Epsilon)
    return 2 * p * r / (p + r + Config.Epsilon)


# arks_normal_cross_entropy = functools.partial(
#     arksKerasTools.proto_weighted_cross_entropy, weights=np.array([[1., 1.]]))
weighted_cross_entropy = arksKerasTools.weighted_cross_entropy


def def_model():
    """
    parameters in configurations.Config
    :returns the model, and the dictionaries
    """
    # input context x
    x_context = keras.layers.Input(shape=[None, ], dtype='int32', name='x_context')
    # event labels
    e_p_context = keras.layers.Input(shape=[None, ], dtype='int32', name='e_p_context')
    e_f_context = keras.layers.Input(shape=[None, ], dtype='int32', name='e_f_context')
    e_p = keras.layers.Input(shape=[1, ], dtype='int32', name='e_p')
    e_f = keras.layers.Input(shape=[1, ], dtype='int32', name='e_f')
    a_t_star = keras.layers.Input(shape=[2, ], dtype='float32', name='a_t_star')
    c_gru_coef = keras.layers.Input(shape=[1, ], dtype='float32', name='c_coef')

    # embeddings
    x_embedding_layer = keras.layers.Embedding(input_dim=Config.Active_vocabulary_size,
                                               output_dim=Config.Input_Embedding_size,
                                               weights=[w2v_matrix], trainable=True)
    event_embedding_layer = keras.layers.Embedding(input_dim=Config.Event_vocabulary_size,
                                                   output_dim=Config.Event_embedding_size)
    x_seq_embedding = x_embedding_layer(x_context)
    e_p_context_embedding = event_embedding_layer(e_p_context)
    e_f_context_embedding = event_embedding_layer(e_f_context)
    e_p_embedding = event_embedding_layer(e_p)
    e_f_embedding = event_embedding_layer(e_f)

    e_f_c = keras.layers.core.Reshape(target_shape=[Config.Event_embedding_size])(e_f_embedding)
    e_p_c = keras.layers.core.Reshape(target_shape=[Config.Event_embedding_size])(e_p_embedding)

    input_to_gru = keras.layers.concatenate([x_seq_embedding, e_p_context_embedding, e_f_context_embedding], axis=2)

    h_t = keras.layers.GRU(Config.RNN_size, dropout=Config.Dropout_rate, name='h_t')(input_to_gru)

    h_t_dropped = keras.layers.Dropout(rate=Config.Dropout_rate)(h_t)

    c_gru = keras.layers.Dense(Config.Event_embedding_size, name='c_gru')(h_t_dropped)

    # determining a
    input_for_a = keras.layers.concatenate([h_t, e_f_c, e_p_c], axis=-1)
    input_for_a_dropped = keras.layers.Dropout(rate=Config.Dropout_rate)(input_for_a)
    # a_hidden = keras.layers.Dense(Config.A_hidden_size, activation='relu', name='a_hidden')(input_for_a_dropped)
    # a_hidden_dropped = keras.layers.core.Dropout(rate=Config.Dropout_rate)(a_hidden)
    # deep a
    # a_t_from_model_output = keras.layers.Dense(2, activation='softmax', name='a')(a_hidden_dropped)
    # shallow model
    a_t_from_model_output = keras.layers.Dense(2, activation='softmax', name='a')(input_for_a_dropped)
    # simplest model
    # a_t_from_model_output = keras.layers.Dense(2, activation='softmax', name='a')(h_t_dropped)

    # linear interpolation
    # a_t_star_2d = keras.layers.core.Reshape(target_shape=(1, 2))(a_t_star)
    # e_gru = e_f_embedding
    # c_gru_2d = keras.layers.core.Reshape(target_shape=(1, Config.Event_embedding_size))(c_gru)
    # ori_candidates = keras.layers.concatenate([c_gru_2d, e_gru], axis=-2)
    # candidates = keras.layers.core.Reshape(target_shape=(2, Config.Event_embedding_size))(ori_candidates)
    # o_t = arksKerasTools.StupidMatrixMultiplicationLayer.create_layer()([a_t_star_2d, candidates])
    # o_t = keras.layers.core.Reshape(target_shape=[Config.Event_embedding_size])(o_t)

    # linear interpolation step 2
    # c_gru_weighted = keras.layers.Multiply()([c_gru_coef, c_gru])
    ###
    # o_t_final = keras.layers.Add()([c_gru_weighted, o_t])

    # o_t_final = c_gru
    o_t_final = c_gru

    input_for_dist = keras.layers.concatenate([o_t_final, e_f_c, e_p_c], axis=-1)
    input_for_dist_dropped = keras.layers.Dropout(rate=Config.Dropout_rate)(input_for_dist)

    output_distribution = keras.layers.Dense(Config.Active_vocabulary_size,
                                             activation='softmax', name='main')(input_for_dist_dropped)

    model = keras.models.Model(inputs=[x_context, e_p_context, e_f_context, e_f, e_p, a_t_star, c_gru_coef],
                               outputs=[output_distribution, a_t_from_model_output],
                               name='model5')

    return model


def train_model(model):
    model.compile(loss={'main': 'categorical_crossentropy',
                        'a': weighted_cross_entropy},
                  optimizer=keras.optimizers.adam(lr=Config.Learning_rate),
                  loss_weights={'main': 1., 'a': Config.Loss_weight_on_a},
                  metrics={'main': ['accuracy'],
                           'a': ['accuracy', f_0, f_1, r_1, p_1]})
    # , corec_1, truth_1, label_1]})

    # callbacks: check point, tensorboard and early stopping
    check_point_path = os.path.join('.\\' + Config.Run_Index, r'{epoch:02d}.hdf5')
    checkpointer = keras.callbacks.ModelCheckpoint(check_point_path,
                                                   monitor='loss', period=1, save_best_only=True, verbose=1)
    csv_logger = keras.callbacks.CSVLogger(Config.CSV_log_path, separator=',', append=False)

    # tensorboard_logger = keras.callbacks.TensorBoard(log_dir=Config.LOG_DIR,
    #                                                 histogram_freq=2, write_grads=True,
    #                                                 write_graph=True, embeddings_freq=0)

    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=Config.Early_stopping_patience)

    sample_generation_callback = SampleGeneration(seed=Config.Seeds[0])

    check_point_on = False
    # tensorboard_on = False

    call_back_list = [csv_logger, early_stopping_callback, sample_generation_callback]

    if check_point_on:
        call_back_list.append(checkpointer)
    # if tensorboard_on:
    #    call_back_list.append(tensorboard_logger)

    model.fit([x_train, e_p_context_train, e_f_context_train, e_f_train, e_p_train, a_train, c_gru_coef_train],
              [y_train, a_train],
              epochs=Config.Max_train_epochs, batch_size=Config.Batch_size,
              validation_data=([x_val, e_p_context_val, e_f_context_val, e_f_val, e_p_val, a_val, c_gru_coef_val],
                               [y_val, a_val]),
              callbacks=call_back_list)

    return model, sample_generation_callback.samples


def load_model():

    # load saved model
    if os.path.exists(Config.Checkpoint_path):
        model_custom_objects = {'weighted_cross_entropy': weighted_cross_entropy,
                                'f_0': f_0, 'f_1': f_1, 'r_1': r_1, 'p_1': p_1}
        model = keras.models.load_model(Config.Checkpoint_path,
                                        custom_objects=model_custom_objects)
        samples = dict()
        return model, samples, True
    else:
        model = def_model()
        model, samples = train_model(model)
        return model, samples, False


def main():
    """
    acquire model and generate 5 stories for each seed in variable 'seed'
    :return:
    """
    # TODO ==================  Train and evaluate model  =========================
    # Config.Seeds = AgendaGenerator.Agenda.generate_seeds(['grocery'], {'grocery': 15})

#    if not os.path.exists(os.path.join('.', Config.Run_Index)):
#        os.mkdir(os.path.join('.', Config.Run_Index))
    # train_samples is empty if the model is loaded from a file.
    # otherwise it records a sample generated after each epoch.
    model, train_samples, loaded_from_checkpoint_file = load_model()

    evaluation = model.evaluate(x=[x_val, e_p_context_val, e_f_context_val, e_f_val, e_p_val, a_val, c_gru_coef_val],
                                y=[y_val, a_val])

    print('eval_trained_model:')
    print(str(evaluation))

    # TODO ===================== Generate Sequences =======================
    scripts = ['grocery']
    seeds_path = 'seeds_binaries_' + Config.Run_Index

    # load seeds
    seeds_dict = dict()
    if os.path.exists(seeds_path):
        seeds_dict = dill.load(open(seeds_path, 'rb'))
    else:
        for script in scripts:
            generation_seeds = list()
            for _ in range(20):
                generation_seeds.append(AgendaGenerator.Agenda.generate_seed(script))
            for _ in range(20):
                generation_seeds.append(AgendaGenerator.Agenda.generate_random_seed(script, length=15))
            seeds_dict[script] = generation_seeds
        dill.dump(seeds_dict, open(seeds_path, 'wb'))

    generation_path = 'generation_binaries_' + Config.Run_Index

    generations = dict()

    # generate texts
    for script in scripts:
        texts, seeds = list(), list()
        for seed in seeds_dict[script]:
            texte, seede = plain_beam(model, seed)
            texts.append(texte), seeds.append(seede)
        generations[script] = texts, seeds

    dill.dump(generations, open(generation_path, 'wb'))

    with open(Config.Sample_path, 'w') as sample_out:
        for script in scripts:
            sample_out.write('=============== Script: ' + script + ' ==============\n')
            tts, sds = generations[script]
            for i, text in enumerate(tts):
                sample_out.write('-- TEXT ' + str(i) + ' ---\n')
                sample_out.write(text + '\n')
        sample_out.write('===============================\n')
        for step, sample in enumerate(train_samples):
            sample_out.write('epoch ' + str(step/2) + ' . Sample:' + sample + '\n')


def random_hyper_search(index_list, opt_log_path):
    header = ','.join(['index', 'dropout', 'lr', 'clipnorm', 'batch_size', 'context_length',
                       'event_embedding_size', 'rnn_size', 'loss on a', 'weight on a1',
                       'main_loss', 'main_acc', 'a_acc', 'af1', 'ar1'])
    if not os.path.exists(opt_log_path):
        with open(opt_log_path, 'w') as log_out:
            log_out.write(header + '\n')

    # dropout, lr, clipnorm, batch_size, context_length, Event_embedding, rnn_size, a loss, a weight
    num_combinations, hyper_lists = utils.load_hypers()
    types = ['c', 'c', 'c', '2e', 'int', '2e', '2e', 'c', 'c']
    for index in index_list:
        print('training with combination ' + str(index))
        # TODO ========== Set Hypers ===========
        hyper_combination = utils.read_hypers(types, hyper_lists, index)
        Config.Dropout_rate = hyper_combination[0]
        Config.Learning_rate_coefficient = hyper_combination[1]
        Config.Clipping_threshold = hyper_combination[2]
        Config.Batch_size = hyper_combination[3]
        Config.Context_length = np.int(hyper_combination[4])
        Config.Event_embedding_size = hyper_combination[5]
        Config.RNN_size = hyper_combination[6]
        Config.Loss_weight_on_a = hyper_combination[7]
        Config.Weight_on_a1_cross_entropy[1] = hyper_combination[8]

        # Train and evaluate model
        model, train_samples, loaded_from_checkpoint_file = load_model()
        evaluation = model.evaluate(
            x=[x_val, e_p_context_val, e_f_context_val, e_f_val, e_p_val, a_val, c_gru_coef_val],
            y=[y_val, a_val])
        with open(opt_log_path, 'a') as log_out:
            # log index, hypers, val loss, p/r on a1.
            # metrics in evaluation by order: loss, main_loss, a_loss, main_acc, a_acc, a_f_0, a_f_1, a_r_1, a_p_1
            # log: main_loss, main_acc, a_acc, af1, ar1
            hypers_str = ','.join([str(param) for param in hyper_combination])
            log_out.write(str(index) + ',' + hypers_str + ',' +
                          str(round(evaluation[1], 3)) + ',' + str(round(evaluation[3], 3)) + ','
                          + str(round(evaluation[4], 3)) + ',' +
                          str(round(evaluation[7], 3)) + ',' + str(round(evaluation[8], 3)) + '\n')


random_hyper_search(list(range(45, 60, 1)), os.path.join('..', 'log_opt2.csv'))

# main()
