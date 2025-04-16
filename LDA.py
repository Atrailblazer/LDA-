import jieba
from gensim import corpora, models
import pandas as pd
import math
import numpy as np
# 读取 Excel 文件
def read_excel(file_path):
    try:
        # 读取 Excel 文件
        df = pd.read_excel(file_path)

        # 获取第一列数据
        first_column = df.iloc[:, 0]  # 使用 iloc 获取第一列
        return first_column.dropna().tolist()  # 返回非空值的列表
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return []

# 中文文本分词
def tokenize(text):
    return list(jieba.cut(text))

# 删除中文停用词
def delete_stopwords(tokens, stop_words):
    return [word for word in tokens if word not in stop_words]

# 移除标点符号
def remove_punctuation(input_string):
    import string
    all_punctuation = string.punctuation + "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏. \t "
    translator = str.maketrans('', '', all_punctuation)
    return input_string.translate(translator)

#计算tf
def tf(doc_words):
    sum_list = list(set(word for word in doc_words))
    words_len = len(doc_words)
    word_dict = {word:0 for word in sum_list}
    for word1 in sum_list:
        for word2 in doc_words:
            if word1 == word2:
                word_dict[word1] += 1
    tf_dict = {}
    for word,count in word_dict.items():
        tf_dict[word] = count/words_len

    # print(tf_dict)
    return tf_dict

# 计算IDF
def idf(doc_list):
    sum_list = list(set([word_i for doc_i in doc_list for word_i in doc_i]))
    idf_dict = {word_i: 0 for word_i in sum_list}
    for word_j in sum_list:
        for doc_j in doc_list:
            if word_j in doc_j:
                idf_dict[word_j] += 1
    return {k: math.log(len(doc_list) / (v + 1)) for k, v in idf_dict.items()}

def tf_idf(tf_dict,idf_dict):
    tfidf = {}
    for word,tf_value in tf_dict.items():
        tfidf[word] = tf_value * idf_dict[word]
    return tfidf
# 主程序
if __name__ == "__main__":
    # 文件路径1
    file_path = r"D:\项目\LDA_data\人民网-新能源1.xlsx"

    # 读取文件
    documents = read_excel(file_path)
    if not documents:
        print("No valid data found.")
        exit()

    # 自定义中文停用词表
    chinese_stopwords = [
        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个",
        "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
        "自己", "这", "我们", "为", "他", "她", "它", "但", "而", "如果", "与", "于",
        "以", "被", "这", "那", "哪个","万", "亿", "从", "等", "使用", "对", "可以",
        "上半年", "拟", "股东","月","中","目前","及","元","减持","年", "北京", "天津",
        "上海", "重庆", "河北", "山西", "辽宁", "吉林", "黑龙江",
        "江苏", "浙江", "安徽", "福建", "江西", "山东", "河南", "湖北", "湖南",
        "广东", "海南", "四川", "贵州", "云南", "陕西", "甘肃", "青海", "台湾",
        "内蒙古", "广西", "西藏", "宁夏", "新疆", "香港", "澳门","板","将","咱们"," ",
        "大","收起","现在","多","今天","就是","日","———", "》）", "）÷（１－", "”，", "）、", "＝（", ":", "→", "℃", "&", "*", "一一", "~~~~", "’", ".", "『", ".一", "./", "--", "』", "＝″", "【", "［＊］", "｝＞",
    "［⑤］］","［①Ｄ］", "ｃ］", "ｎｇ昉", "＊", "//", "［", "］", "［②ｅ］", "［②ｇ］", "＝｛", "}", "也", "‘", "Ａ", "［①⑥］", "［②Ｂ］", "［①ａ］", "［④ａ］",
    "［①③］", "［③ｈ］", "③］", "１．", "－－", "［②ｂ］", "’‘", "×××", "［①⑧］", "０：２", "＝［", "［⑤ｂ］", "［②ｃ］", "［④ｂ］", "［②③］", "［③ａ］", "［④ｃ］",
    "［①⑤］", "［①⑦］", "［①ｇ］", "∈［", "［①⑨］", "［①④］", "［①ｃ］", "［②ｆ］", "［②⑧］", "［②①］", "［①Ｃ］", "［③ｃ］", "［③ｇ］", "［②⑤］", "［②②］", "一.",
    "［①ｈ］", ".数", "［］", "［①Ｂ］", "数/", "［①ｉ］", "［③ｅ］", "［①①］", "［④ｄ］", "［④ｅ］", "［③ｂ］", "［⑤ａ］", "［①Ａ］", "［②⑧］", "［②⑦］", "［①ｄ］",
    "［②ｊ］", "〕〔", "］［", "://", "′∈", "［②④", "［⑤ｅ］", "１２％", "ｂ］", "...", "...................", "…………………………………………………③", "ＺＸＦＩＴＬ", "［③Ｆ］", "」",
    "［①ｏ］", "］∧′＝［", "∪φ∈", "′｜", "｛－", "②ｃ", "｝", "［③①］", "Ｒ．Ｌ．", "［①Ｅ］", "Ψ", "－［＊］－", "↑", ".日", "［②ｄ］", "［②", "［②⑦］", "［②②］",
    "［③ｅ］", "［①ｉ］", "［①Ｂ］", "［①ｈ］", "［①ｄ］", "［①ｇ］", "［①②］", "［②ａ］", "ｆ］", "［⑩］", "ａ］", "［①ｅ］", "［②ｈ］", "［②⑥］", "［③ｄ］", "［②⑩］",
    "ｅ］", "〉", "】", "元／吨", "［②⑩］", "２．３％", "５：０", "［①］", "::", "［②］", "［③］", "［④］", "［⑤］", "［⑥］", "［⑦］", "［⑧］", "［⑨］", "……", "——", "?", "、", "。",
    "“", "”", "《", "》", "！", "，", "：", "；", "？", "．", ",", "．", "'", "? ", "·", "———", "──", "? ", "—", "<", ">", "（", "）", "〔", "〕", "[", "]", "(", ")", "-", "+", "～", "×",
    "／", "/", "①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩", "Ⅲ", "В", '"', ";", "#", "@", "γ", "μ", "φ", "φ．", "×", "Δ", "■", "▲", "sub", "exp", "sup", "sub", "Lex", "＃", "％",
    "＆", "＇", "＋", "＋ξ", "＋＋", "－", "－β", "＜", "＜±", "＜Δ", "＜λ", "＜φ", "＜＜", "=", "＝", "＝☆", "＝－", "＞", "＞λ", "＿", "～±", "～＋", "≈", "②Ｇ", "－［＊］－", ".....", "〉",
    "③⑩", "......","第二", "一番", "一直", "一个", "一些", "许多", "种", "有的是", "也就是说", "末##末", "啊", "阿", "哎", "哎呀", "哎哟", "唉", "俺", "俺们", "按", "按照", "吧", "吧哒", "把", "罢了", "被",
    "本", "本着", "比", "比方", "比如", "鄙人", "彼", "彼此", "边", "别", "别的", "别说", "并", "并且", "不比", "不成", "不单", "不但", "不独", "不管", "不光", "不过", "不仅", "不拘",
    "不论", "不怕", "不然", "不如", "不特", "不惟", "不问", "不只", "朝", "朝着", "趁", "趁着", "乘", "冲", "除", "除此之外", "除非", "除了", "此", "此间", "此外", "从", "从而", "打",
    "待", "但", "但是", "当", "当着", "到", "得", "的", "的话", "等", "等等", "地", "第", "叮咚", "对", "对于", "多", "多少", "而", "而况", "而且", "而是", "而外", "而言", "而已",
    "尔后", "反过来", "反过来说", "反之", "非但", "非徒", "否则", "嘎", "嘎登", "该", "赶", "个", "各", "各个", "各位", "各种", "各自", "给", "根据", "跟", "故", "故此", "固然", "关于",
    "管", "归", "果然", "果真", "过", "哈", "哈哈", "呵", "和", "何", "何处", "何况", "何时", "嘿", "哼", "哼唷", "呼哧", "乎", "哗", "还是", "还有", "换句话说", "换言之", "或",
    "或是", "或者", "极了", "及", "及其", "及至", "即", "即便", "即或", "即令", "即若", "即使", "几", "几时", "己", "既", "既然", "既是", "继而", "加之", "假如", "假若", "假使",
    "鉴于", "将", "较", "较之", "叫", "接着", "结果", "借", "紧接着", "进而", "尽", "尽管", "经", "经过", "就", "就是", "就是说", "据", "具体地说", "具体说来", "开始", "开外",
    "靠", "咳", "可", "可见", "可是", "可以", "况且", "啦", "来", "来着", "离", "例如", "哩", "连", "连同", "两者", "了", "临", "另", "另外", "另一方面", "论", "嘛", "吗", "慢说",
    "漫说", "冒", "么", "每", "每当", "们", "莫若", "某", "某个", "某些", "拿", "哪", "哪边", "哪儿", "哪个", "哪里", "哪年", "哪怕", "哪天", "哪些", "哪样", "那", "那边", "那儿",
    "那个", "那会儿", "那里", "那么", "那么些", "那么样", "那时", "那些", "那样", "乃", "乃至", "呢", "能", "你", "你们", "您", "宁", "宁可", "宁肯", "宁愿", "哦", "呕", "啪达",
    "旁人", "呸", "凭", "凭借", "其", "其次", "其二", "其他", "其它", "其一", "其余", "其中", "起", "起见", "岂但", "恰恰相反", "前后", "前者", "且", "然而", "然后", "然则",
    "让", "人家", "任", "任何", "任凭", "如", "如此", "如果", "如何", "如其", "如若", "如上所述", "若", "若非", "若是", "啥", "上下", "尚且", "设若", "设使", "甚而", "甚么",
    "甚至", "省得", "时候", "什么", "什么样", "使得", "是", "是的", "首先", "谁", "谁知", "顺", "顺着", "似的", "虽", "虽然", "虽说", "虽则", "随", "随着", "所", "所以", "他",
    "他们", "他人", "它", "它们", "她", "她们", "倘", "倘或", "倘然", "倘若", "倘使", "腾", "替", "通过", "同", "同时", "哇", "万一", "往", "望", "为", "为何", "为了", "为什么",
    "为着", "喂", "嗡嗡", "我", "我们", "呜", "呜呼", "乌乎", "无论", "无宁", "毋宁", "嘻", "吓", "相对而言", "像", "向", "向着", "嘘", "呀", "焉", "沿", "沿着", "要", "要不",
    "要不然", "要么", "要是", "也", "也罢", "也好", "一", "一般", "一旦", "一方面", "一来", "一切", "一样", "一则", "依", "依照", "矣", "以", "以便", "以及", "以免", "以至",
    "以至于", "以致", "抑或", "因", "因此", "因而", "因为", "哟", "用", "由", "由此可见", "由于", "有", "有的", "有关", "有些", "又", "于", "于是", "于是乎", "与", "与此同时",
    "与否", "与其", "越是", "云云", "哉", "再说", "再者", "在", "在下", "咱", "咱们", "则", "怎", "怎么", "怎么办", "怎么样", "怎样", "咋", "照", "照着", "者", "这", "这边",
    "这儿", "这个", "这会儿", "这就是说", "这里", "这么", "这么点儿", "这么些", "这么样", "这时", "这些", "这样", "正如", "吱", "之", "之类", "之所以", "之一", "只是", "只限",
    "只要", "只有", "至", "至于", "诸位", "着", "着呢", "自", "自从", "自个儿", "自各儿", "自己", "自家", "自身", "综上所述", "总的来看", "总的来说", "总的说来", "总而言之",
    "总之", "纵", "纵令", "纵然", "纵使", "遵照", "作为", "兮", "呃", "呗", "咚", "咦", "喏", "啐", "喔唷", "嗬", "嗯", "嗳","进行","全","时间","主任","天","后","计入","进入",
    "继续","全","只","昨天","点","共","股及","链接","还","客","以上","可达","相关","做","大家","已经","没","不是"
    ]
    english_letters_stopwords = [
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p",
        "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P",
        "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
    ]
    numbers_stopwords = [
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
    ]
    custom_stop_words = chinese_stopwords+english_letters_stopwords+numbers_stopwords  # 示例停用词表

    # 文本清洗和分词
    cleaned_texts = []
    tf_list = []
    # tf_idf_list = []
    documents_single = {""}
    for document in documents:
        # 去除停用词
        no_punct = remove_punctuation(document)
        documents_single.add(no_punct)

    for document in documents_single:
        # 去除停用词
        # no_punct = remove_punctuation(document)

        # print(no_punct)
        # 分词
        tokens = tokenize(document)


        # print(tokens)
        filtered_tokens = delete_stopwords(tokens, custom_stop_words)

        tf_list.append(tf(filtered_tokens))

        cleaned_texts.append(filtered_tokens)

    # print(cleaned_texts)
    print(len(cleaned_texts))
    idf_dict = idf(cleaned_texts)
    # print(idf_dict)

    tfidf_list = []
    for tf_dict in tf_list:
        tfidf_list.append(tf_idf(tf_dict,idf_dict))

    # sum = 0
    max_weight_list = []
    for weight_dict in tfidf_list:
        max_weight = 0
        for word,weight in weight_dict.items():
            max_weight = max(max_weight,weight)
        max_weight_list.append(max_weight)
        # sum += max_weight


    means = np.mean(max_weight_list)
    std_weight = np.std(max_weight_list)
    threshold = means-0.5*std_weight

    # print(means)
    # print(std_weight)
    # print(threshold)

    data = []
    for weight_dict in tfidf_list:
        max_weight = 0
        words = []
        for word,weight in weight_dict.items():
            words.append(word)
            max_weight = max(max_weight,weight)
        # max_weight_list.append(max_weight)
        if max_weight > threshold:
            data.append(words)
        # else:
        #     print(words)

    print(len(data))
    print(data)
    # print(tfidf_list)

    # 创建词典和语料库
    dictionary = corpora.Dictionary(cleaned_texts)
    # print(dictionary)




    corpus = [dictionary.doc2bow(text) for text in cleaned_texts]
    # print(corpus)
'''
    # 训练 LDA 模型
    lda = models.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=50)

    # 打印主题
    for idx, topic in lda.print_topics(-1):
        print(f"主题 {idx + 1}：{topic}")

    # from sklearn.feature_extraction.text import CountVectorizer
    #
    #
    # # N-gram 抽取函数
    # def extract_ngrams(cleaned_texts, ngram_range=(2, 4)):
    #     # 将清洗后的分词结果转回字符串
    #     documents_as_text = [" ".join(text) for text in cleaned_texts]
    #     # 使用 CountVectorizer 提取 N-gram
    #     vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words=None)
    #     X = vectorizer.fit_transform(documents_as_text)
    #     feature_names = vectorizer.get_feature_names_out()
    #     return feature_names, X
    #
    #
    # # 添加 N-gram 提取到主程序流程中
    # if cleaned_texts:
    #     ngram_range = (2, 4)  # 自定义 N-gram 范围
    #     feature_names, X = extract_ngrams(cleaned_texts, ngram_range)
    #
    #     # 显示部分结果
    #     print(f"提取的 N-gram 特征名称（部分）：{feature_names[:10]}")
    #     print(f"词频矩阵形状：{X.shape}")
'''