from django.shortcuts import render
import pandas as pd
import pickle

category_data = pd.read_csv("idx2category.csv")
idx2category = {row.k: row.v for idx,row in category_data.iterrows()} #辞書型に変換

#モデルの読込
with open("rdmf.pickle", mode="rb") as f:
    model = pickle.load(f)

def index(request):
    #GETのみでアクセスされた場合
    if request.method == "GET":
        return render(
            request,
            "nlp/home.html"
        )
    #POSTがある場合
    else:
        #home.htmlの"title"という名前のinputタグに入力された値をリストで取得
        title = [request.POST["title"]] 
        print("title:", title)
        result = model.predict(title)[0]
        print("result:", result)
        pred = idx2category[result]
        return render(
            request,
            "nlp/home.html",
            {"title":pred} #htmlでtitle変数を{{title}}で取得できるようにする
        )

