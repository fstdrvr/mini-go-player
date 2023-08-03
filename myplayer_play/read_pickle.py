import pickle
#
# with open('qtable.pickle', 'rb') as handle:
#     q_values = pickle.load(handle)
# with open('hist_states.pickle', 'rb') as handle:
#     history_states = pickle.load(handle)
#
# print(len(q_values))
# print(history_states)

# import pickle
# q_values = {}
# history_states = []
# with open('qtable.pickle', 'wb') as handle:
#     pickle.dump(q_values, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('hist_states.pickle', 'wb') as handle:
#     pickle.dump(history_states, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# print(q_values)
# print(history_states)

import collections

def _global_helper_function():
    return [0, 0]

# trans_table = collections.defaultdict(_global_helper_function)
#
# with open('trans_table.pickle', 'wb') as handle:
#     pickle.dump(trans_table, handle, protocol=min(pickle.HIGHEST_PROTOCOL, 4))

with open('trans_table.pickle', 'rb') as handle:
    trans_table = pickle.load(handle)
print(len(trans_table))
print(trans_table)