import numpy as np
import cv2

# クエリ画像をquery_imgとして読み込む
# and train image このクエリ画像
# このクエリ画像は訓練画像から探す必要があるものである
# 同じディレクトリに保存する
# image.jpgという名前で保存する  

query_img = cv2.imread('query.png')
train_img = cv2.imread('train.png')
 
# グレースケールに変換する
query_img_bw = cv2.cvtColor(query_img,cv2.COLOR_BGR2GRAY)
train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)
 
# ORB検出アルゴリズムを初期化する
orb = cv2.ORB_create()
 
# 次に，キーポイントを検出し，クエリ画像の # ディスクリプタを計算する．
# クエリ画像のディスクリプタを計算する
# および学習画像
queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_bw, None)
trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw, None)

# マッチャーの初期化
# キーポイントをマッチングし
# キーポイントにマッチする
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(queryDescriptors,trainDescriptors)
# マッチをソートする
matches = sorted(matches, key=lambda x: x.distance)

# マッチを最終画像に描画する
# 両方の画像を含む drawMatches()
# 関数は両方の画像とキーポイントを受け取り
final_img = cv2.drawMatches(query_img_bw, queryKeypoints, 
train_img_bw, trainKeypoints, matches[:20], None, 
flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, matchColor=(0,0,2550))


# 画像を保存する
cv2.imwrite("matches.png", final_img)
