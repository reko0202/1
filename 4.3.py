#一个stramlit应用
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

pdf_file = st.file_uploader("上传PDF文件", type=["pdf"])

if pdf_file is not None:
    file_details = {"文件名": pdf_file.name, "文件类型": pdf_file.type, "文件大小": pdf_file.size}
    st.write(file_details)

    try:
        pdf = fitz.open(stream=pdf_file.read(), filetype="pdf")
        page = pdf.load_page(0)
        text = page.get_text()
        st.write(text)
    except:
        st.error("无法解析该文件。")

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

 st.line_chart(list(zip(x, y)))
