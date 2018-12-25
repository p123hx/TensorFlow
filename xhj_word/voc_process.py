import jieba
import codecs
import collections
from operator import itemgetter

RAW ="/Users/cql/Desktop/xhj_word/xhj_a_s"
OUPUT="xhj_a.vocab"

counter=collections.Counter()
with codecs.open(RAW, "r", "utf-8") as f:
    for line in f:
        seg_list = jieba.cut(line, cut_all=False)
        for _ in seg_list:
            counter[_] +=1
sorted_word_to_cnt =sorted(counter.items(),
                           key=itemgetter(1),
                           reverse=True)
sorted_words = [x[0] for x in sorted_word_to_cnt]
sorted_words = ["<unk>","<sos>","<eos>"]+sorted_words
with codecs.open(OUPUT,"w","utf-8") as file_output:
    for word in sorted_words:
        file_output.write(word + "\n")

