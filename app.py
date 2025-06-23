import streamlit as st
from openai import OpenAI
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from snownlp import SnowNLP
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger
import json

# 📌 CKIP 斷詞器與詞性標註器（初始化一次即可）
ws = CkipWordSegmenter(model="bert-base")
pos = CkipPosTagger(model="bert-base")

# 📌 頁面設定
st.set_page_config(page_title="中文陳述分析系統", layout="wide")
st.title("📊 中文陳述主題與詞性分析系統")

# 📌 輸入 OpenAI API Key
api_key = st.text_input("請輸入你的 OpenAI API Key", type="password")
client = None
if api_key:
    client = OpenAI(api_key=api_key)

# 📌 輸入陳述資料與主題數目
statement_text = st.text_area("請貼上你的中文陳述資料", height=300)
num_topics = st.number_input("請輸入要分幾個主題（至少 3 個）", min_value=3, value=3)

# 📌 定義 LLM 分析函式
def analyze_statement_to_timeline(statement, num_topics):
    prompt = f"""
你是一位陳述資料分析師，以下是一份中文陳述資料。
請根據內容與時間順序，將資料歸類為 {num_topics} 個主題。
每個主題請自動命名，並列出該主題下依照時序的事件摘要。

⚠️ 回覆時只提供符合範例格式的 JSON 字串，勿加任何說明文字。

回覆格式：
[
  {{
    "主題": "主題名稱",
    "事件列表": [
      "時間 + 事件內容",
      ...
    ]
  }},
  ...
]

陳述資料：
{statement}
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content

# 📌 按鈕觸發分析
if st.button("開始分析"):
    if not api_key or not statement_text:
        st.warning("請確認已輸入 API Key 與陳述資料")
    else:
        with st.spinner("分析中，請稍候..."):
            try:
                # GPT 分析
                result_text = analyze_statement_to_timeline(statement_text, num_topics).strip()
                if result_text.startswith("```json"):
                    result_text = result_text[7:]
                if result_text.endswith("```"):
                    result_text = result_text[:-3]

                result_data = json.loads(result_text)

                # 整理時序表
                rows = []
                for topic in result_data:
                    for event in topic['事件列表']:
                        rows.append({"主題": topic['主題'], "事件": event})

                df = pd.DataFrame(rows)
                st.subheader("🗂️ 時序表")
                st.dataframe(df, use_container_width=True)

                # 主題統計表
                topic_summary = df.groupby('主題')['事件'].apply(lambda x: ' '.join(x)).reset_index()
                topic_summary['總字數'] = topic_summary['事件'].apply(len)
                topic_summary['總詞數'] = topic_summary['事件'].apply(lambda x: len(SnowNLP(x).words))

                st.subheader("📑 主題-文字、字數、詞數統計表")
                st.dataframe(topic_summary, use_container_width=True)

                # 互動表格
                fig_table = go.Figure(data=[go.Table(
                    header=dict(values=list(topic_summary.columns),
                                fill_color='paleturquoise',
                                align='left'),
                    cells=dict(values=[topic_summary[c] for c in topic_summary.columns],
                               fill_color='lavender',
                               align='left'))
                ])
                st.plotly_chart(fig_table, use_container_width=True)

                # 字數折線圖
                fig_line = go.Figure()
                fig_line.add_trace(go.Scatter(
                    x=topic_summary['主題'],
                    y=topic_summary['總字數'],
                    mode='lines+markers',
                    line=dict(color='royalblue')
                ))
                fig_line.update_layout(title='每個主題總字數折線圖',
                                       xaxis_title='主題',
                                       yaxis_title='總字數')
                st.plotly_chart(fig_line, use_container_width=True)

                # 詞數折線圖
                fig_wordcount = go.Figure()
                fig_wordcount.add_trace(go.Scatter(
                    x=topic_summary['主題'],
                    y=topic_summary['總詞數'],
                    mode='lines+markers',
                    line=dict(color='orange')
                ))
                fig_wordcount.update_layout(title='每個主題總詞數折線圖',
                                            xaxis_title='主題',
                                            yaxis_title='總詞數')
                st.plotly_chart(fig_wordcount, use_container_width=True)

                # CKIP 斷詞 + 詞性標註
                word_segments = ws([statement_text])
                pos_tags = pos(word_segments)
                words = word_segments[0]
                tags = pos_tags[0]
                df_pos = pd.DataFrame({"詞": words, "詞性": tags})

                st.subheader("📚 詞性分布統計表")
                pos_count = df_pos["詞性"].value_counts().reset_index()
                pos_count.columns = ["詞性", "數量"]
                st.dataframe(pos_count, use_container_width=True)

                # 詞性直方圖
                st.subheader("📊 詞性分布直方圖")
                fig_pos = plt.figure(figsize=(12, 6))
                plt.bar(pos_count["詞性"], pos_count["數量"], color='skyblue')
                plt.xlabel("詞性")
                plt.ylabel("數量")
                plt.title("詞性分布統計圖")
                plt.xticks(rotation=45)
                st.pyplot(fig_pos)

            except Exception as e:
                st.error(f"分析發生錯誤：{e}")

