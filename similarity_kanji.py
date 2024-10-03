import fasttext
from sklearn.metrics.pairwise import cosine_similarity

gz_model_path = 'cc.ja.300.bin'  # 自分のモデルファイルのパスに変更

# モデルファイルをfastTextでロード
model = fasttext.load_model(gz_model_path)

# コサイン類似度を計算する関数
def cosine_similarity_between_kanji(kanji1, kanji2):
    # 漢字のベクトルを取得
    vector1 = model.get_word_vector(kanji1)
    vector2 = model.get_word_vector(kanji2)

    # コサイン類似度の計算
    cos_sim = cosine_similarity([vector1], [vector2])

    return cos_sim[0][0]

# サンプルの漢字ペア
kanji1 = '猿'
kanji2 = '猫'

# 類似度の計算
similarity = cosine_similarity_between_kanji(kanji1, kanji2)
print(f"コサイン類似度({kanji1}, {kanji2}): {similarity:.4f}")

