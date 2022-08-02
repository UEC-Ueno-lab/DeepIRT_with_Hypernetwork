import numpy as np
from utils import getLogger
# Code reused from https://github.com/ckyeungac/DeepIRT.git

# reuse the code from DeepIRT

class DataLoader():
    def __init__(self, n_questions, n_skills, seq_len, separate_char):
        self.separate_char = separate_char
        self.n_questions = n_questions
        self.n_skills = n_skills
        self.seq_len = seq_len

    # the dataloader for dataset with both item and skill tag
    def load_data(self, path, mode):
        s_data = []
        q_item_data = []
        qa_data = []
        with open(path, 'r') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                # skip the number of sequence
                if line_idx % 4 == 0:
                    continue
                # handle question_line
                elif line_idx % 4 == 1:
                    q_item_tag_list_1 = line.split(self.separate_char)
                    if len(q_item_tag_list_1) > 0:
                        q_item_tag_list = q_item_tag_list_1
                        q_item_tag_list = [s for s in q_item_tag_list if s != '']
                    else:
                        continue
                # handle answer-line
                elif line_idx % 4 == 2:
                    s_tag_list_1 = line.split(self.separate_char)
                    if len(s_tag_list_1) > 0:
                        s_tag_list = s_tag_list_1
                        s_tag_list = [s for s in s_tag_list if s != '']
                    else:
                        continue
                elif line_idx % 4 == 3:
                    a_tag_list_1 = line.split(self.separate_char)
                    if len(a_tag_list_1) > 0:
                        a_tag_list = a_tag_list_1
                        a_tag_list = [s for s in a_tag_list if s != '']
                        n_split = len(q_item_tag_list) // self.seq_len
                        if len(q_item_tag_list) % self.seq_len != 0:
                            n_split += 1
                        for k in range(n_split):
                            # temporary container for each sequence
                            s_container = list()
                            q_item_container = list()
                            qa_container = list()
                            s_true_container = list()

                            start_idx = k * self.seq_len
                            end_idx = min((k + 1) * self.seq_len, len(a_tag_list), len(s_tag_list),
                                          len(q_item_tag_list))
                            for i in range(start_idx, end_idx):
                                q_item_value = int(q_item_tag_list[i])
                                a_value = int(a_tag_list[i])  # either be 0 or 1
                                s_value = int(s_tag_list[i])
                                if 'assist2017_akt' in path:
                                  qa_value = q_item_value + a_value * self.n_skills
                                else:
                                  qa_value = s_value + a_value * self.n_skills
                                s_container.append(s_value)
                                q_item_container.append(q_item_value)
                                qa_container.append(qa_value)
                            s_data.append(s_container)
                            q_item_data.append(q_item_container)
                            qa_data.append(qa_container)

                    else:
                        continue

        # convert it to numpy array
        s_data_array = np.zeros((len(s_data), self.seq_len))
        q_item_data_array = np.zeros((len(q_item_data), self.seq_len))
        qa_data_array = np.zeros((len(qa_data), self.seq_len))
        for i in range(len(q_item_data)):
            _s_data = s_data[i]
            _q_item_data = q_item_data[i]
            _qa_data = qa_data[i]
            s_data_array[i, :len(_s_data)] = _s_data
            q_item_data_array[i, :len(_q_item_data)] = _q_item_data
            qa_data_array[i, :len(_qa_data)] = _qa_data
        if 'assist2017_akt' in path:
          return s_data_array , q_item_data_array,qa_data_array
        else:
          return q_item_data_array, s_data_array, qa_data_array

    # the dataloader for dataset with skill tag
    def load_data_skill(self, path):
        q_data = []
        qa_data = []
        with open(path, 'r') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                # skip the number of sequence
                if line_idx % 3 == 0:
                    continue
                # handle question_line
                elif line_idx % 3 == 1:
                    q_tag_list = line.split(self.separate_char)
                    q_tag_list_int = [int(num) for num in q_tag_list]
                # handle answer-line
                elif line_idx % 3 == 2:
                    a_tag_list = line.split(self.separate_char)
                    a_tag_list = [x.strip() for x in a_tag_list if x.strip() != '']
                    q_tag_list = [x.strip() for x in q_tag_list if x.strip() != '']
                    # find the number of split for this sequence
                    n_split = len(q_tag_list) // self.seq_len
                    if len(q_tag_list) % self.seq_len != 0:
                        n_split += 1
                    for k in range(n_split):
                        # temporary container for each sequence
                        q_container = list()
                        qa_container = list()
                        start_idx = k * self.seq_len
                        end_idx = min((k + 1) * self.seq_len, len(a_tag_list), len(q_tag_list))

                        for i in range(start_idx, end_idx):
                            q_value = int(q_tag_list[i])
                            a_value = int(a_tag_list[i])  # either be 0 or 1
                            qa_value = q_value + a_value * self.n_questions
                            q_container.append(q_value)
                            qa_container.append(qa_value)
                        q_data.append(q_container)
                        qa_data.append(qa_container)
        # convert it to numpy array
        q_data_array = np.zeros((len(q_data), self.seq_len))
        qa_data_array = np.zeros((len(q_data), self.seq_len))
        for i in range(len(q_data)):
            _q_data = q_data[i]
            _qa_data = qa_data[i]
            q_data_array[i, :len(_q_data)] = _q_data
            qa_data_array[i, :len(_qa_data)] = _qa_data

        return q_data_array, qa_data_array
