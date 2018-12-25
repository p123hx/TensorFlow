import jieba
import codecs
RAW ="/Users/cql/Desktop/xhj_word/xhj_a_s"
OUTPUT_DATA="xhj_a_train"
VOCAB="xhj_a.vocab"
with codecs.open(VOCAB, "r", "utf-8") as f_vocab:
    vocab= [w.strip() for w in f_vocab.readlines()]
word_to_id = {k: v for (k,v) in zip(vocab, range(len(vocab)))}

def get_id(word):
    return word_to_id[word] if word in word_to_id else word_to_id["<unk>"]
fin = codecs.open(RAW, "r", "utf-8")
fout = codecs.open(OUTPUT_DATA, "w", "utf-8")
for line in fin:
    seg_list = jieba.lcut(line,cut_all=False)+["<eos>"]
    out_line=' '.join([str(get_id(w)) for w in seg_list]) + "\n"
    fout.write(out_line)
fin.close()
fout.close()
