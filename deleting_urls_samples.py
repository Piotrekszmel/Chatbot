from chatbot_helper import clean_text

with open('train.from', 'r', encoding='utf-8') as f:
    train_Q = f.read().split('\n')

Q_idxs = []
for i, line in enumerate(train_Q):
    train_Q[i] = clean_text(line)
    for word in line.split(' '):
        if word.startswith('http') or word.startswith('https'):             #looking for samples with urls
            Q_idxs.append(i)


with open('/home/pszmelcz/Downloads/train.to', 'r', encoding='utf-8') as f:
    train_A = f.read().split('\n')
A_idxs = []
for i, line in enumerate(train_A):
    train_A[i] = clean_text(line)
    for word in line.split(' '):
        if word.startswith('http') or word.startswith('https'):             #looking for samples with urls
            A_idxs.append(i)

same_idx = []
if len(Q_idxs) > 0:
    for idx in sorted(Q_idxs, reverse=True):
        if idx in A_idxs:
            same_idx.append(idx)
        del train_Q[idx]
        del train_A[idx]                           
        A_idxs[:] = [index - 1 if index > idx else index for index in A_idxs]       #removing samples with urls from questions and answers
if len(A_idxs) > 0:
    for idx in sorted(A_idxs, reverse=True):
        if idx not in same_idx:
            del train_Q[idx]
            del train_A[idx]


with open('Questions.txt', 'w', encoding='utf-8') as f:
    for line in train_Q:
        f.write(line + '\n')

with open('Answers.txt', 'w', encoding='utf-8') as f:
    for line in train_A:
        f.write(line + '\n')

with open('Answers.txt', 'r', encoding='utf-8') as f:
    A = f.read().split('\n')
with open('Questions.txt', 'r', encoding='utf-8') as f:
    Q = f.read().split('\n')

Q_and_A = []
for i in range(len(Q)):
    Q_and_A.append(Q[i])
    Q_and_A.append(A[i])    

with open('train_data.txt', 'w', encoding='utf-8') as f:        #saving questions and answers list
    for i, line in enumerate(Q_and_A):
       f.write(line + '\n')