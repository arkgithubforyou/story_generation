from configurations import Config

import collections
import keras
import numpy as np
import os
import gensim
import dill
import re


def model_6_loader(script_list=Config.Effective_Scripts,
                   val_proportion=Config.Validation_proportion,
                   courpus_folder=Config.Clean_corpus_6_folder):
    """
    loads data for specified scripts
    each execution returns different splits, but the dictionaries are same
    text tokens converted to lowercase
    :returns
    x: [seq_len, None]
    e_p: [seq_len, None]
    e_f: [seq_len, None]
    effective previous / forthcoming events for each point in x.
    u: [seq_len, 1]
    utterance updates. is 1 at event words, or 1/n if the event utterance is longer than 1.
    a: [2, None] selectional coefficients
    y(one-hot): [Voc_size, None]
    dictionaries: word2int, int2word, event2int, int2event
    """

    # load corpus
    corpus = list()
    text = list()
    events = list()
    for script in script_list:
        with open(os.path.join(courpus_folder, script), 'r') as fin:
            inline = fin.readline()
            while inline != '':
                splited_inline = inline.split()
                splited_inline[0] = splited_inline[0].lower()
                corpus.append(splited_inline)
                text.append(splited_inline[0])
                events.extend(splited_inline[1:])
                inline = fin.readline()

    # build dictionaries
    token_counter = collections.Counter(text)
    event_counter = collections.Counter(events)

    token_counter = token_counter.most_common(Config.Max_vocabulary_size)
    dict_word_to_int = dict()
    dict_word_to_int[Config.Padding_token] = 0
    for word, count in token_counter:
        if count >= Config.Least_frequency or Config.Concatenation_delimiter in word:
            dict_word_to_int[word] = len(dict_word_to_int)
    dict_int_to_word = dict(zip(dict_word_to_int.values(), dict_word_to_int.keys()))

    event_counter = event_counter.most_common(Config.Max_vocabulary_size)
    dict_event_to_int = dict()
    for word, _ in event_counter:
        dict_event_to_int[word] = len(dict_event_to_int)
    for sc in Config.Effective_Scripts:
        dict_event_to_int[Config.Beginning_Event + '_' + sc] = len(dict_event_to_int)
        dict_event_to_int[Config.Ending_Event + '_' + sc] = len(dict_event_to_int)
    dict_int_to_event = dict(zip(dict_event_to_int.values(), dict_event_to_int.keys()))

    # set vocabulary size back
    Config.Active_vocabulary_size = len(dict_word_to_int)
    # to include beginning and ending events in the event list
    Config.Event_vocabulary_size = len(dict_event_to_int)
    print('vocabulary size:', len(token_counter), 'kept:', Config.Active_vocabulary_size,
          'text tokens:', len(text), 'events:', Config.Event_vocabulary_size)

    stories = list()
    tmp_story = list()
    for i in range(len(corpus)):
        tmp_story.append(corpus[i])
        if corpus[i][0].find(Config.End_of_story_label) != -1:
            stories.append(tmp_story)
            tmp_story = list()

    x = list()
    y = list()
    a = list()
    e_p = list()
    e_f = list()
    e_p_context = list()
    e_f_context = list()

    # note: story does not contain 'End of story' token.
    for story in stories:
        # pad data. lol this is convenient
        for index in range(Config.Context_length - 1):
            story.insert(0, [Config.Padding_token])
        # prepare data in desired form
        current_script = story[-1][0][15:]
        story.pop()
        x_temp = list()
        a_temp = list()
        y_temp = list()
        e_p_temp = list()
        e_f_temp = list()
        e_p_context_temp = list()
        e_f_context_temp = list()

        e_p_story = list()
        e_f_story = list()

        # determine event labels
        for pointer in range(len(story)):
            # determine previous event
            if len(story[pointer]) > 1:
                cache = story[pointer][1]
            else:
                cache = ''
            i = pointer - 1
            label = False
            while i >= 0:
                if len(story[i]) > 1:
                    cache_prime = story[i][1]
                else:
                    cache_prime = ''
                if cache_prime != cache and cache_prime != '':
                    e_p_story.append(cache_prime)
                    label = True
                    break
                i -= 1
            if not label:
                e_p_story.append(Config.Beginning_Event + '_' + current_script)
            # determine forthcoming event
            if len(story[pointer]) > 1:
                e_f_story.append(story[pointer][1])
            else:
                label = False
                i = pointer + 1
                while i < len(story):
                    if len(story[i]) > 1:
                        e_f_story.append(story[i][1])
                        label = True
                        break
                    i += 1
                if not label:
                    e_f_story.append(Config.Ending_Event + '_' + current_script)

        # collect data
        x_pointer = Config.Context_length
        while x_pointer < len(story):
            # collect context
            context = list()
            for i in range(Config.Context_length):
                context.append(story[x_pointer - Config.Context_length + i][0])
            x_temp.append(context)
            # y
            y_temp.append(story[x_pointer][0])
            # a
            if len(story[x_pointer]) > 1:
                a_temp.append([0, 1])
            else:
                a_temp.append([1, 0])
            # collect context events
            context_ep = list()
            context_ef = list()
            for i in range(Config.Context_length):
                context_ep.append(e_p_story[x_pointer - Config.Context_length + i])
                context_ef.append(e_f_story[x_pointer - Config.Context_length + i])
            e_p_context_temp.append(context_ep)
            e_f_context_temp.append(context_ef)
            # determine previous event
            e_p_temp.append(e_p_story[x_pointer])
            # determine forthcoming event
            e_f_temp.append(e_f_story[x_pointer])

            x_pointer += 1

        x.extend(x_temp)
        y.extend(y_temp)
        e_p.extend(e_p_temp)
        e_f.extend(e_f_temp)
        e_p_context.extend(e_p_context_temp)
        e_f_context.extend(e_f_context_temp)
        a.extend(a_temp)

    # convert to integers
    x_int = np.zeros(np.shape(x))
    y_int = np.zeros(np.shape(y))
    a_int = np.array(a)
    e_p_int = np.zeros(np.shape(e_p))
    e_f_int = np.zeros(np.shape(e_f))
    e_p_context_int = np.zeros(np.shape(e_p_context))
    e_f_context_int = np.zeros(np.shape(e_f_context))
    batch_size = np.shape(x)[0]
    for i in range(batch_size):
        for j in range(Config.Context_length):
            if x[i][j] in dict_word_to_int:
                x_int[i, j] = dict_word_to_int[x[i][j]]
            else:
                x_int[i, j] = dict_word_to_int['<unk>']
            e_p_context_int[i, j] = dict_event_to_int[e_p_context[i][j]]
            e_f_context_int[i, j] = dict_event_to_int[e_f_context[i][j]]
        if y[i] in dict_word_to_int:
            y_int[i] = dict_word_to_int[y[i]]
        else:
            y_int[i] = dict_word_to_int['<unk>']
        e_p_int[i] = dict_event_to_int[e_p[i]]
        e_f_int[i] = dict_event_to_int[e_f[i]]

    y_int = keras.utils.to_categorical(y_int, num_classes=Config.Active_vocabulary_size)

    # split data
    # shuffle

    perm = np.random.permutation(len(x_int))
#    x_int, y_int, e_p_int, e_f_int, e_p_context_int, e_f_context_int, a_int = \
#        x_int[perm], y_int[perm], e_p_int[perm], e_f_int[perm], \
#        e_p_context_int[perm], e_f_context_int[perm], a_int[perm]

#    x_train = x_int[perm[:int(batch_size * (1 - val_proportion))]]
    x_train = x_int[perm[:int(batch_size * (1 - val_proportion))]]
    y_train = y_int[perm[:int(batch_size * (1 - val_proportion))]]
    e_p_train = e_p_int[perm[:int(batch_size * (1 - val_proportion))]]
    e_f_train = e_f_int[perm[:int(batch_size * (1 - val_proportion))]]
    e_p_context_train = e_p_context_int[perm[:int(batch_size * (1 - val_proportion))]]
    e_f_context_train = e_f_context_int[perm[:int(batch_size * (1 - val_proportion))]]
    a_train = a_int[perm[:int(batch_size * (1 - val_proportion))]]

    x_val = x_int[perm[int(batch_size * (1 - val_proportion)):]]
    y_val = y_int[perm[int(batch_size * (1 - val_proportion)):]]
    e_p_val = e_p_int[perm[int(batch_size * (1 - val_proportion)):]]
    e_f_val = e_f_int[perm[int(batch_size * (1 - val_proportion)):]]
    e_p_context_val = e_p_context_int[perm[int(batch_size * (1 - val_proportion)):]]
    e_f_context_val = e_f_context_int[perm[int(batch_size * (1 - val_proportion)):]]
    a_val = a_int[perm[int(batch_size * (1 - val_proportion)):]]

    return x_train, y_train, e_p_train, e_f_train, e_p_context_train, e_f_context_train, a_train, \
        x_val, y_val, e_p_val, e_f_val, e_p_context_val, e_f_context_val, a_val, \
        dict_int_to_word, dict_word_to_int, dict_int_to_event, dict_event_to_int, stories


def model_6_wash(foler_input=Config.Clean_corpus_folder, foler_output=Config.Clean_corpus_6_folder):
    """
    wash data:
    model_3:
    remove event labels Unrel and RelNScr
    replace most punktuations with '.'
    post-fix Evoking events with script names
    remove redundant event labels
    model_4:
    concatenate event tokens
    model_6:
    Unrels and RelNScrs also post-fixed with script names.
    no longer replaces punktuations with '.'
    model_7:
    events are nolong merged.
    """
    for script in Config.All_Scripts:
        corpus = list()
        text = list()
        events = list()
        with open(os.path.join(foler_input, script), 'r') as fin:
            inline = fin.readline()
            while inline != '':
                splited_inline = inline.split()
                splited_inline[0] = splited_inline[0].lower()
                corpus.append(splited_inline)
                text.append(splited_inline[0])
                events.extend(splited_inline[1:])
                inline = fin.readline()
        with open(os.path.join(foler_output, script), 'w') as fout:
            i = 0
            while i < len(corpus):
                line = corpus[i]
                # probe event length
                # this is safe: the last token is <end_of_story> without event label.
                span = 1
                if len(line) > 1:
                    while len(corpus[i + span]) > 1:
                        if line[1] != corpus[i + span][1] or line[1] in Config.Invalid_Labels:
                            break
                        span += 1

                token = ''
                for j in range(span):
                    if j == 0:
                        token += corpus[i + j][0]
                    else:
                        token += Config.Concatenation_delimiter + corpus[i + j][0]

                label = ''
                if len(line) > 1:
                    label = line[1]
                if label in Config.Invalid_Labels:
                    label = ''
                if label in Config.Post_fix_events:
                    label += '_' + script
                if token == Config.End_of_story_label:
                    token += '_' + script
                if re.fullmatch('[0-9.]+', token) is not None and token != '.':
                    token = Config.Universal_number
                fout.write(token + ' ' + label + '\n')
                i += span


def int_to_word(seq, dict_int_to_word):
    """
    convert simple sequence of ints to text
    """
    return [dict_int_to_word[seq[i]] for i in range(len(seq))]


def word_to_int(seq, dict_word_to_int):
    """
    convert list of tokens sentence to list of ints
    """
    return [dict_word_to_int[seq[i]] for i in range(len(seq))]


def word2vec_wrapper(dw2i, w2v_original_path=Config.Word2Vec_Google_Path, w2v_save_path=Config.W2V_wrapper_path):
    """
    wrappes w2v vectors to a dill file.
    format: dict word_index -> word_vec
    concatenated tokens takes the average of all its components as the vector
    :param dw2i:
    :return:
    """
    w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_original_path, binary=True)
    print('w2v loaded.')
    vocabulary_size = len(dw2i)
    embedding_matrix = np.ndarray(shape=[vocabulary_size, 300])
    for word, index in dw2i.items():
        if word.find(Config.Concatenation_delimiter) != -1 or word.find('-') != -1:
            word_replaced = word.replace(Config.Concatenation_delimiter, '-')
            words = word_replaced.split('-')
            embedding = np.zeros([300])
            for word_piece in words:
                if word_piece in w2v.wv:
                    embedding += w2v.wv[word_piece]
                else:
                    print(word_piece + ' not found')
                    embedding += np.random.normal(0, 0.1, size=[300])
            embedding_matrix[index, :] = embedding / len(words)
        else:
            if word in w2v.wv:
                embedding_matrix[index, :] = w2v.wv[word]
            else:
                embedding_matrix[index, :] = np.random.normal(0, 0.1, size=[300])
                print(word + ' not found2')
    print('oorah!')

    dill.dump(embedding_matrix, open(w2v_save_path, 'wb'))


def load_word2vec_matrix(w2v_path=Config.W2V_wrapper_path):
    return dill.load(open(w2v_path, 'rb'))


def seq_generator(script_list=list(['bath']), courpus_folder=Config.Clean_corpus_6_folder):
    # load corpus
    corpus = list()
    text = list()
    events = list()
    for script in script_list:
        with open(os.path.join(courpus_folder, script), 'r') as fin:
            inline = fin.readline()
            while inline != '':
                splited_inline = inline.split()
                splited_inline[0] = splited_inline[0].lower()
                corpus.append(splited_inline)
                text.append(splited_inline[0])
                events.extend(splited_inline[1:])
                inline = fin.readline()

    # build dictionaries
    token_counter = collections.Counter(text)
    event_counter = collections.Counter(events)

    token_counter = token_counter.most_common(Config.Max_vocabulary_size)
    dict_word_to_int = dict()
    dict_word_to_int[Config.Padding_token] = 0
    for word, count in token_counter:
        if count >= Config.Least_frequency or Config.Concatenation_delimiter in word:
            dict_word_to_int[word] = len(dict_word_to_int)
    dict_int_to_word = dict(zip(dict_word_to_int.values(), dict_word_to_int.keys()))

    event_counter = event_counter.most_common(Config.Max_vocabulary_size)
    dict_event_to_int = dict()
    for word, _ in event_counter:
        dict_event_to_int[word] = len(dict_event_to_int)
    for sc in Config.Effective_Scripts:
        dict_event_to_int[Config.Beginning_Event + '_' + sc] = len(dict_event_to_int)
        dict_event_to_int[Config.Ending_Event + '_' + sc] = len(dict_event_to_int)
    dict_int_to_event = dict(zip(dict_event_to_int.values(), dict_event_to_int.keys()))

    # set vocabulary size back
    Config.Active_vocabulary_size = len(dict_word_to_int)
    # to include beginning and ending events in the event list
    Config.Event_vocabulary_size = len(dict_event_to_int)
    print('vocabulary size:', len(token_counter), 'kept:', Config.Active_vocabulary_size,
          'text tokens:', len(text), 'events:', Config.Event_vocabulary_size)

    stories = list()
    tmp_story = list()
    for i in range(len(corpus)):
        tmp_story.append(corpus[i])
        if corpus[i][0].find(Config.End_of_story_label) != -1:
            stories.append(tmp_story)
            tmp_story = list()

    t2close = 0
    close2t = 0
    for story in stories:
        events=[s[1] for s in story if len(s) > 1]
        if 'ScrEv_check_temp' in events and 'ScrEv_close_drain' in events:
            tind = events.index('ScrEv_check_temp')
            uind = events.index('ScrEv_close_drain')

            if tind <= uind:
                t2close+=1
            else:
                close2t+=1
    ss = 1
    pass


if __name__ == '__main__':
    # seq_generator()
    # model_6_wash()
    x_train, y_train, e_p_train, e_f_train, e_p_context_train, e_f_context_train, a_train,  \
        x_val, y_val, e_p_val, e_f_val, e_p_context_val, e_f_context_val, a_val,  \
        di2w, dw2i, di2e, de2i = model_6_loader()
    # word2vec_wrapper(dw2i)
    pass
