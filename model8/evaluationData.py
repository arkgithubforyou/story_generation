import configurations
import dataLoaders
import AgendaGenerator
import dill
import os


class DataGenerator(object):
    scripts = ['bath', 'bicycle', 'bus', 'cake', 'flight', 'grocery', 'haircut', 'library', 'train', 'tree']
    script_descriptions = {'bath': 'taking a bath',
                           'bicycle': 'fixing a flat tire',
                           'bus': 'taking a bus',
                           'cake': 'baking a cake',
                           'flight': 'taking a flight',
                           'grocery': 'grocery shopping',
                           'haircut': 'doing a haircut',
                           'library': 'borrowing a book from library',
                           'train': 'taking a train',
                           'tree': 'planting a tree'}

    @staticmethod
    def generate_data(texts, seeds, delimiter='\t', output_file='exp_data.csv'):
        """
        takes text (in string), seed pairs to generate experiment data.
        Fields to generate:
            itemID, text, script, scriptDescription = ing-form

            Feed options:
            Include
            Irrelevant
            narration
            Agenda: ['', '', '']

        :param texts:
        :param seeds:
        :param delimiter:
        :return:
        """
        with open(output_file, 'w') as fout:
            for i, text in enumerate(texts):
                seed = seeds[i]
                script = seed.script
                script_description = DataGenerator.script_descriptions[script]
                fout.write(str(i) + delimiter)
                fout.write(text + delimiter)
                fout.write(script + delimiter)
                fout.write(script_description + delimiter)

                agenda_unique = list()
                for evente in seed.agenda:
                    if evente in de2d:
                        if de2d[evente] not in agenda_unique:
                            agenda_unique.append(de2d[evente])

                fout.write('\"' + str(agenda_unique) + '\"' + delimiter)
                inclusion_opt, relevance_opt, narration_opt = list(), list(), list()

                inclusion_opt.append(
                    'A. [Good] It mentions as many necessary steps as a real person would when telling a story about '
                    + script_description + '.')
                inclusion_opt.append(
                    'B. [OK] Some but not many crucial steps are missing, the story is still ok as a whole.')
                inclusion_opt.append(
                    'C. [Not OK] Many crucial steps are missing. The text is nowhere close to a complete story about '
                    + script_description + '.')
                inclusion_opt.append(
                    'D. [Bad] Too many crucial steps are missing. It is not even clear that this is a story about '
                    + script_description + '.')
                fout.write('\"' + str(inclusion_opt) + '\"' + delimiter)

                relevance_opt.append(
                    'A. [Not at All] All contents of the text are clearly relevant.')
                relevance_opt.append(
                    'B. [One or Two] One or two irrelevant activities are mentioned. The story is still ok as a whole.')
                relevance_opt.append(
                    'C. [Quite a Few] The story mentions quite a few activities irrelevant to '
                    + script_description + '.')
                relevance_opt.append(
                    'D. [Too Many] The story is a complete mess. It is not even clear that the story is about '
                    + script_description + '.')
                fout.write('\"' + str(relevance_opt) + '\"' + delimiter)

                narration_opt.append(
                    'A. [Good] The text mentioned the activities in a plausible order as they may happen in reality.')
                narration_opt.append(
                    'B. [OK] One or two activities are not described at plausible points of time. '
                    'Most activities mentioned are still ok if we consider the story as a whole.')
                narration_opt.append(
                    'C. [Bad] The order of the activities is a complete mess. There is absolutely no way that  '
                    + script_description
                    + ' could happen as is described.')
                fout.write('\"' + str(narration_opt) + '\"\n')

    @staticmethod
    def generate_data_from_story(story, de2d):
        """
        generate seed from story and feed to generate_data
        add 'story_begin_script' to the seed
        :param story:
        :return:
        """
        script = ''
        agenda = list()
        for token in story:
            if len(token) > 1:
                if token[1].find('voking') != -1:
                    script = [sc for sc in DataGenerator.scripts if token[1].find(sc) != -1][0]
                    break
        agenda.append('Story_Begin_' + script)
        labels = [token[1] for token in story if len(token) > 1]
        agenda.extend([label for label in labels if label in de2d])
        text = ' '.join([token[0] for token in story if token[0].find('<') == -1]).replace('+', ' ')
        seed = configurations.Config.Seed([], [], [], agenda)
        return text, seed

    @staticmethod
    def generate_data_from_text(text, script):
        seed = AgendaGenerator.Agenda.generate_random_seed(script, length=15)
        return text, seed

# seeds = configurations.Config.Seeds[:1]
# texts = ['this is a story .']
# DataGenerator.generate_data(texts, seeds, output_file='tt.csv')


def load_event_descriptions(path=configurations.Config.Event_descriptions):
    de2d1, dd2e1 = dict(), dict()
    with open(path, 'r') as fin:
        buf = fin.readline()
        while buf != '':
            buf2 = fin.readline()
            de2d1[buf[:-1]] = buf2[:-1]
            dd2e1[buf2[:-1]] = buf[:-1]
            buf = fin.readline()

    return dd2e1, de2d1


x_train, y_train, e_p_train, e_f_train, e_p_context_train, e_f_context_train, a_train, \
    x_val, y_val, e_p_val, e_f_val, e_p_context_val, e_f_context_val, a_val, \
    di2w, dw2i, di2e, de2i, stories = dataLoaders.model_6_loader()

dd2e, de2d = load_event_descriptions()


# arrange stories into a dict according to the scripts
story_dict = dict()
for story in stories:
    script = ''
    for token in story:
        if len(token) > 1:
            if token[1].find('voking') != -1:
                script = [sc for sc in DataGenerator.scripts if token[1].find(sc) != -1][0]
                break
    if script not in story_dict:
        story_dict[script] = list()
    story_dict[script].append(story)


# TODO:===============================   Generation   =====================================
print('generating exp data...')

GRU_dill_path = 'gru_samples_dict'
# GRU dill: dict script -> story_list

Model8_dill_path = 'samples_binary_' + configurations.Config.Run_Index
# model8 dill: texts, seeds. previously generated.
# 4 texts for each seed, 2 random + 2 normal seeds for each script

scripts = ['grocery']
indices = {'grocery': [0, 2]}

for script in scripts:
    for index in indices[script]:
        Output_path = os.path.join('eval_data', 'data_' + script + str(index) + '.csv')
        texts, seeds = list(), list()

        # this yields single story from original text
        texte, seed = DataGenerator.generate_data_from_story(story_dict[script][index], de2d)
        texts.append(texte), seeds.append(seed)

        # gru stories
        # now 4 for each script
        GRU_dict = dill.load(open(GRU_dill_path, 'rb'))
        gru_text = ' '.join(GRU_dict[script][index])
        gru_text, gru_seed = DataGenerator.generate_data_from_text(gru_text, script)
        texts.append(gru_text), seeds.append(gru_seed)

        # generations: random and rational
        m8texts, m8seeds = dill.load(open(Model8_dill_path, 'rb'))
        texts.append(m8texts[index])
        seeds.append(m8seeds[index])
        texts.append(m8texts[index + 10])
        seeds.append(m8seeds[index + 10])

        DataGenerator.generate_data(texts, seeds, output_file=Output_path)


a = 1

# with open(configurations.Config.event_descriptions, 'w') as fout:
#     for event in de2i:
#         fout.write(event + '\n')
