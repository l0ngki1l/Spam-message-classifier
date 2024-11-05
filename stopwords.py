def read_stopwords(stopwords_path):
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        stopwords = f.read()
    stopwords = stopwords.splitlines()
    return stopwords


if __name__ == '__main__':
    print("请输入停用词库的路径")
    path = input()
    stopwords = read_stopwords(path)
    print(stopwords[-20:])
