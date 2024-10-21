import fasttext
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

gz_model_path = 'cc.ja.300.bin'  # 自分のモデルファイルのパスに変更
filename = 'dat.csv'

# モデルファイルをfastTextでロード
model = fasttext.load_model(gz_model_path)

# コサイン類似度を計算する関数
def cosine_similarity_between_kanji(kanji1, kanji2):
    # 漢字のベクトルを取得
    vector1 = model.get_word_vector(kanji1)
    vector2 = model.get_word_vector(kanji2)

    # コサイン類似度の計算
    cos_sim = cosine_similarity([vector1], [vector2])
#    print(cos_sim)

    return cos_sim[0][0]

# ファイルから漢字読み込み（ファイルの形式やフィールド名が変わると要修正）
df = pd.read_csv(filename)
list_kanji = df["Japanese"].tolist()

# 総当たりでコサイン類似度を計算
n = len(list_kanji)
similarity_matrix = np.zeros((n, n))

for i in range(n):
   for j in range(n):
      similarity_matrix[i][j] = cosine_similarity_between_kanji(list_kanji[i], list_kanji[j])

df = pd.DataFrame(similarity_matrix, index=list_kanji, columns=list_kanji)
df.to_csv('output_kanji_similarity.csv')
