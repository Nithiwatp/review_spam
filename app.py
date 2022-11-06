# main website content
import streamlit as st
import numpy as np
import joblib

@st.cache
def predict(review):
    cv_th = joblib.load('./models/cv_th_bow.h5')
    x = cv_th.transform([review]) 
    y_pred_prob = model.predict_proba(x)
    return y_pred_prob

# setup page
CURRENT_THEME = "dark"
IS_DARK_THEME = True
st.set_page_config(page_title='Spam Detection', page_icon="📝")
st.set_option('deprecation.showfileUploaderEncoding', False) # disable deprecation error
with open("app.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

st.header("AI that filter the spam review.")

# for loading model
@st.cache(allow_output_mutation = True)     # enable cache to improve the loading time
def get_model():
    model = joblib.load('./models/xg_bow.h5')
    return model

with st.spinner('Loading Model...'):
    model = get_model()

# text input
review = st.text_area('Review',placeholder ="Input the review...",height = 300)

if review is not None:
  if st.button('Predict'):
    prediction = predict(review)
    classes = np.argmax(prediction, axis = 1)
    if classes is not None:
      if classes == 1:
          result = "I'm " + str(np.round(np.max(prediction)*100,2)) +"% sure that this is SPAM."
      else:
          result = "I'm " + str(np.round(np.max(prediction)*100,2)) +"% sure that this is HAM."
      st.subheader(result)
