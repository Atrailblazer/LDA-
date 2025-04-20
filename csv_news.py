import numpy as np
import pandas as pd
import jieba
import re
import math
from typing import List, Dict, Tuple
import string
from multiprocessing import Pool, cpu_count  # 添加导入

# --------------------------
# 常量定义
# --------------------------
TARGET_COLUMNS = [
    "作者", "Author full names", "作者 ID", "文献标题", "年份",
    "来源出版物名称", "卷", "期", "论文编号", "起始页码",
    "结束页码", "页码计数", "施引文献", "DOI", "链接",
    "摘要", "作者关键字", "索引关键字", "文献类型",
    "出版阶段", "开放获取", "来源出版物", "EID"
]

# --------------------------
# 并行分词优化
# --------------------------
def initialize_jieba():
    """初始化jieba分词器"""
    jieba.initialize()
    jieba.enable_parallel(cpu_count())  # 使用正确的cpu_count

def parallel_tokenize(texts: List[str], stopwords: List[str]) -> List[List[str]]:
    """并行分词处理"""
    with Pool(processes=cpu_count()) as pool:  # 使用正确的cpu_count
        results = pool.starmap(_tokenize_worker, [(text, stopwords) for text in texts])
    return results

def _tokenize_worker(text: str, stopwords: List[str]) -> List[str]:
    """单进程分词工作函数"""
    try:
        text = re.sub(r'[\W_]+', '', str(text))  # 更快的去标点方法
        words = jieba.lcut(text, cut_all=False)
        return [w for w in words if w not in stopwords and len(w) > 1]
    except Exception as e:
        print(f"分词出错: {e}")
        return []

# --------------------------
# 文本处理模块
# --------------------------
def load_stopwords() -> List[str]:
    """加载自定义停用词表"""
    chinese_stopwords = [
        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个",
        "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
        "自己", "这", "我们", "为", "他", "她", "它", "但", "而", "如果", "与", "于",
        "以", "被", "这", "那", "哪个", "万", "亿", "从", "等", "使用", "对", "可以",
        "上半年", "拟", "股东", "月", "中", "目前", "及", "元", "减持", "年", "北京", "天津",
        "上海", "重庆", "河北", "山西", "辽宁", "吉林", "黑龙江",
        "江苏", "浙江", "安徽", "福建", "江西", "山东", "河南", "湖北", "湖南",
        "广东", "海南", "四川", "贵州", "云南", "陕西", "甘肃", "青海", "台湾",
        "内蒙古", "广西", "西藏", "宁夏", "新疆", "香港", "澳门", "板", "将", "咱们", " ",
        "大", "收起", "现在", "多", "今天", "就是", "日", "———", "》）", "）÷（１－", "”，", "）、", "＝（", ":", "→", "℃", "&",
        "*", "一一", "~~~~", "’", ".", "『", ".一", "./", "--", "』", "＝″", "【", "［＊］", "｝＞",
        "［⑤］］", "［①Ｄ］", "ｃ］", "ｎｇ昉", "＊", "//", "［", "］", "［②ｅ］", "［②ｇ］", "＝｛", "}", "也", "‘", "Ａ", "［①⑥］", "［②Ｂ］",
        "［①ａ］", "［④ａ］",
        "［①③］", "［③ｈ］", "③］", "１．", "－－", "［②ｂ］", "’‘", "×××", "［①⑧］", "０：２", "＝［", "［⑤ｂ］", "［②ｃ］", "［④ｂ］", "［②③］",
        "［③ａ］", "［④ｃ］",
        "［①⑤］", "［①⑦］", "［①ｇ］", "∈［", "［①⑨］", "［①④］", "［①ｃ］", "［②ｆ］", "［②⑧］", "［②①］", "［①Ｃ］", "［③ｃ］", "［③ｇ］", "［②⑤］",
        "［②②］", "一.",
        "［①ｈ］", ".数", "［］", "［①Ｂ］", "数/", "［①ｉ］", "［③ｅ］", "［①①］", "［④ｄ］", "［④ｅ］", "［③ｂ］", "［⑤ａ］", "［①Ａ］", "［②⑧］",
        "［②⑦］", "［①ｄ］",
        "［②ｊ］", "〕〔", "］［", "://", "′∈", "［②④", "［⑤ｅ］", "１２％", "ｂ］", "...", "...................",
        "…………………………………………………③", "ＺＸＦＩＴＬ", "［③Ｆ］", "」",
        "［①ｏ］", "］∧′＝［", "∪φ∈", "′｜", "｛－", "②ｃ", "｝", "［③①］", "Ｒ．Ｌ．", "［①Ｅ］", "Ψ", "－［＊］－", "↑", ".日", "［②ｄ］", "［②",
        "［②⑦］", "［②②］",
        "［③ｅ］", "［①ｉ］", "［①Ｂ］", "［①ｈ］", "［①ｄ］", "［①ｇ］", "［①②］", "［②ａ］", "ｆ］", "［⑩］", "ａ］", "［①ｅ］", "［②ｈ］", "［②⑥］",
        "［③ｄ］", "［②⑩］",
        "ｅ］", "〉", "】", "元／吨", "［②⑩］", "２．３％", "５：０", "［①］", "::", "［②］", "［③］", "［④］", "［⑤］", "［⑥］", "［⑦］", "［⑧］",
        "［⑨］", "……", "——", "?", "、", "。",
        "“", "”", "《", "》", "！", "，", "：", "；", "？", "．", ",", "．", "'", "? ", "·", "———", "──", "? ", "—", "<", ">",
        "（", "）", "〔", "〕", "[", "]", "(", ")", "-", "+", "～", "×",
        "／", "/", "①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩", "Ⅲ", "В", '"', ";", "#", "@", "γ", "μ", "φ", "φ．",
        "×", "Δ", "■", "▲", "sub", "exp", "sup", "sub", "Lex", "＃", "％",
        "＆", "＇", "＋", "＋ξ", "＋＋", "－", "－β", "＜", "＜±", "＜Δ", "＜λ", "＜φ", "＜＜", "=", "＝", "＝☆", "＝－", "＞", "＞λ", "＿",
        "～±", "～＋", "≈", "②Ｇ", "－［＊］－", ".....", "〉",
        "③⑩", "......", "第二", "一番", "一直", "一个", "一些", "许多", "种", "有的是", "也就是说", "末##末", "啊",
        "阿", "哎", "哎呀", "哎哟", "唉", "俺", "俺们", "按", "按照", "吧", "吧哒", "把", "罢了", "被",
        "本", "本着", "比", "比方", "比如", "鄙人", "彼", "彼此", "边", "别", "别的", "别说", "并", "并且", "不比",
        "不成", "不单", "不但", "不独", "不管", "不光", "不过", "不仅", "不拘",
        "不论", "不怕", "不然", "不如", "不特", "不惟", "不问", "不只", "朝", "朝着", "趁", "趁着", "乘", "冲", "除",
        "除此之外", "除非", "除了", "此", "此间", "此外", "从", "从而", "打",
        "待", "但", "但是", "当", "当着", "到", "得", "的", "的话", "等", "等等", "地", "第", "叮咚", "对", "对于",
        "多", "多少", "而", "而况", "而且", "而是", "而外", "而言", "而已",
        "尔后", "反过来", "反过来说", "反之", "非但", "非徒", "否则", "嘎", "嘎登", "该", "赶", "个", "各", "各个",
        "各位", "各种", "各自", "给", "根据", "跟", "故", "故此", "固然", "关于",
        "管", "归", "果然", "果真", "过", "哈", "哈哈", "呵", "和", "何", "何处", "何况", "何时", "嘿", "哼", "哼唷",
        "呼哧", "乎", "哗", "还是", "还有", "换句话说", "换言之", "或",
        "或是", "或者", "极了", "及", "及其", "及至", "即", "即便", "即或", "即令", "即若", "即使", "几", "几时", "己",
        "既", "既然", "既是", "继而", "加之", "假如", "假若", "假使",
        "鉴于", "将", "较", "较之", "叫", "接着", "结果", "借", "紧接着", "进而", "尽", "尽管", "经", "经过", "就",
        "就是", "就是说", "据", "具体地说", "具体说来", "开始", "开外",
        "靠", "咳", "可", "可见", "可是", "可以", "况且", "啦", "来", "来着", "离", "例如", "哩", "连", "连同", "两者",
        "了", "临", "另", "另外", "另一方面", "论", "嘛", "吗", "慢说",
        "漫说", "冒", "么", "每", "每当", "们", "莫若", "某", "某个", "某些", "拿", "哪", "哪边", "哪儿", "哪个",
        "哪里", "哪年", "哪怕", "哪天", "哪些", "哪样", "那", "那边", "那儿",
        "那个", "那会儿", "那里", "那么", "那么些", "那么样", "那时", "那些", "那样", "乃", "乃至", "呢", "能", "你",
        "你们", "您", "宁", "宁可", "宁肯", "宁愿", "哦", "呕", "啪达",
        "旁人", "呸", "凭", "凭借", "其", "其次", "其二", "其他", "其它", "其一", "其余", "其中", "起", "起见", "岂但",
        "恰恰相反", "前后", "前者", "且", "然而", "然后", "然则",
        "让", "人家", "任", "任何", "任凭", "如", "如此", "如果", "如何", "如其", "如若", "如上所述", "若", "若非",
        "若是", "啥", "上下", "尚且", "设若", "设使", "甚而", "甚么",
        "甚至", "省得", "时候", "什么", "什么样", "使得", "是", "是的", "首先", "谁", "谁知", "顺", "顺着", "似的",
        "虽", "虽然", "虽说", "虽则", "随", "随着", "所", "所以", "他",
        "他们", "他人", "它", "它们", "她", "她们", "倘", "倘或", "倘然", "倘若", "倘使", "腾", "替", "通过", "同",
        "同时", "哇", "万一", "往", "望", "为", "为何", "为了", "为什么",
        "为着", "喂", "嗡嗡", "我", "我们", "呜", "呜呼", "乌乎", "无论", "无宁", "毋宁", "嘻", "吓", "相对而言", "像",
        "向", "向着", "嘘", "呀", "焉", "沿", "沿着", "要", "要不",
        "要不然", "要么", "要是", "也", "也罢", "也好", "一", "一般", "一旦", "一方面", "一来", "一切", "一样", "一则",
        "依", "依照", "矣", "以", "以便", "以及", "以免", "以至",
        "以至于", "以致", "抑或", "因", "因此", "因而", "因为", "哟", "用", "由", "由此可见", "由于", "有", "有的",
        "有关", "有些", "又", "于", "于是", "于是乎", "与", "与此同时",
        "与否", "与其", "越是", "云云", "哉", "再说", "再者", "在", "在下", "咱", "咱们", "则", "怎", "怎么", "怎么办",
        "怎么样", "怎样", "咋", "照", "照着", "者", "这", "这边",
        "这儿", "这个", "这会儿", "这就是说", "这里", "这么", "这么点儿", "这么些", "这么样", "这时", "这些", "这样",
        "正如", "吱", "之", "之类", "之所以", "之一", "只是", "只限",
        "只要", "只有", "至", "至于", "诸位", "着", "着呢", "自", "自从", "自个儿", "自各儿", "自己", "自家", "自身",
        "综上所述", "总的来看", "总的来说", "总的说来", "总而言之",
        "总之", "纵", "纵令", "纵然", "纵使", "遵照", "作为", "兮", "呃", "呗", "咚", "咦", "喏", "啐", "喔唷", "嗬",
        "嗯", "嗳", "进行", "全", "时间", "主任", "天", "后", "计入", "进入",
        "继续", "全", "只", "昨天", "点", "共", "股及", "链接", "还", "客", "以上", "可达", "相关", "做", "大家",
        "已经", "没", "不是", '\u3000', '\n', "更", "成为", "里", "部门"
    ]
    english_stopwords = [
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p",
        "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P",
        "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
    ]
    number_stopwords = [str(i) for i in range(1000000)]
    return list(set(chinese_stopwords + english_stopwords + number_stopwords))


def remove_punctuation(text: str) -> str:
    """移除标点符号"""
    punctuation = string.punctuation + "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏. \t "
    return text.translate(str.maketrans('', '', punctuation))


def tokenize(text: str) -> List[str]:
    """中文分词"""
    return list(jieba.cut(text))


def preprocess_text(text: str, stopwords: List[str]) -> List[str]:
    """文本预处理流水线"""
    text = remove_punctuation(text)
    tokens = tokenize(text)
    print(tokens)
    return [word for word in tokens if word not in stopwords and len(word) > 1]


# --------------------------
# 日期处理模块
# --------------------------
def process_dates(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """处理日期列并添加年份列"""
    df[date_col] = pd.to_datetime(df[date_col], format='%Y年%m月%d日', errors='coerce')
    df['年份'] = df[date_col].dt.month.apply(
        lambda x: 2023 if x <= 4 else (2024 if x <= 8 else 2025)
    )
    return df.drop(columns=[date_col])


# --------------------------
# TF-IDF计算模块
# --------------------------
def compute_tf(doc_words: List[str]) -> Dict[str, float]:
    """计算词频(TF)"""
    word_counts = {}
    for word in doc_words:
        word_counts[word] = word_counts.get(word, 0) + 1
    return {word: count / len(doc_words) for word, count in word_counts.items()}


def compute_idf(documents: List[List[str]]) -> Dict[str, float]:
    """计算逆文档频率(IDF)"""
    doc_count = len(documents)
    word_doc_count = {}

    for doc in documents:
        for word in set(doc):
            word_doc_count[word] = word_doc_count.get(word, 0) + 1

    return {word: math.log(doc_count / (count + 1)) for word, count in word_doc_count.items()}


def compute_tfidf(tf_dict: Dict[str, float], idf_dict: Dict[str, float]) -> Dict[str, float]:
    """计算TF-IDF"""
    return {word: tf_value * idf_dict.get(word, 0) for word, tf_value in tf_dict.items()}


# --------------------------
# 主流程
# --------------------------
def load_and_preprocess(filepath: str) -> Tuple[pd.DataFrame, List[List[str]]]:
    """优化版数据预处理"""
    initialize_jieba()  # 必须先初始化

    df = pd.read_excel(filepath)
    df = process_dates(df, '日期')
    stopwords = load_stopwords()

    # 并行处理
    texts = df['索引关键字'].astype(str).tolist()
    cleaned_texts = parallel_tokenize(texts, stopwords)

    # 更新原始列
    df['索引关键字'] = [';'.join(tokens) for tokens in cleaned_texts]

    return df, cleaned_texts


def calculate_weights(documents: List[List[str]]) -> List[Dict[str, float]]:
    """计算TF-IDF权重"""
    # 计算所有文档的TF
    tf_list = [compute_tf(doc) for doc in documents]

    # 计算IDF
    idf_dict = compute_idf(documents)

    # 计算TF-IDF
    return [compute_tfidf(tf_dict, idf_dict) for tf_dict in tf_list]


def filter_and_save(df: pd.DataFrame,
                    cleaned_texts: List[List[str]],
                    tfidf_list: List[Dict[str, float]],
                    output_path: str):
    """筛选数据并保存结果（简化版）"""
    # 计算权重阈值
    max_weights = [max(tfidf.values()) if tfidf else 0 for tfidf in tfidf_list]
    threshold = np.mean(max_weights) - 0.5 * np.std(max_weights)

    # 添加权重列并筛选
    df['权重'] = max_weights
    filtered_df = df[df['权重'] > threshold].copy()

    # 确保输出包含所有目标列
    result_df = pd.DataFrame(columns=TARGET_COLUMNS)
    for col in TARGET_COLUMNS:
        if col in filtered_df.columns:
            result_df[col] = filtered_df[col]
        else:
            result_df[col] = ""  # 填充空值

    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')


def main():
    # 配置参数
    input_file = "D:\\项目\\LDA_data\\data\\test.xlsx"
    output_file = "D:\\项目\\LDA_data\\data\\test_data.csv"

    # 1. 加载并预处理数据
    print("正在加载和预处理数据...")
    df, cleaned_texts = load_and_preprocess(input_file)

    # 2. 计算TF-IDF权重
    print("正在计算TF-IDF权重...")
    tfidf_list = calculate_weights(cleaned_texts)
    print(tfidf_list)

    # 3. 筛选并保存结果
    print("正在筛选和保存结果...")
    filter_and_save(df, cleaned_texts, tfidf_list, output_file)


    print(f"处理完成，结果已保存到 {output_file}")


if __name__ == "__main__":
    main()