from io import TextIOWrapper
import sys
import streamlit as st


st.set_page_config(page_title='Pneumonia Detection', page_icon='favicon.ico',
                   layout='centered')


try:
    from PIL import Image
    import numpy
    from webapp_predict import predict_vgg16, pred_no_session
except:
    st.error('Some error occurred while importing the modules!')

st.title('Pneumonia Detector')

st.subheader(
    "A simple application to detect if a person has pneumonia or not by looking at chest xray")

uploaded_xray = st.file_uploader(
    'Upload your X-ray image:', type=['jpg', 'png', 'jpeg'], caption="Uploaded X-ray")

if uploaded_xray is None:
    st.text("You haven't uploaded an image yet.")
    st.write('Example:')
    st.error('Pneumonia Detected')
    st.image('person67_virus_126.jpeg', use_column_width=True)
else:
    try:
        name = TextIOWrapper(uploaded_xray)
        print('Someone Uploaded:', name.name)
        image = Image.open(uploaded_xray)
        image.save(f'predict/uploads/{name.name}')
        result = predict_vgg16()

        if result == 0:
            st.success('Normal')
        else:
            st.error('Pneumonia Detected')
        st.image(image, use_column_width=True)

    except ValueError:
        st.error(
            'Some error occurred! Please try uploading the X-ray image again or contact piyushbatra1999')
        print("***Exception Occurred:", sys.exc_info())
    except:
        print("***Exception Occurred:", sys.exc_info())
        st.error('Some error occurred! Please Contact piyushbatra1999')

st.warning('This is just an initial POC for pediatric pneumonia detection. The prediction model deployed on this POC website does not produce the exact same results mentioned in the paper. This is due to the fact that this web server does not have a GPU and memory resources.')
st.markdown('<p>Created by <a href="http://piyushbatra.com/" target="_blank" rel="noopener">Piyush Batra</a></p>', unsafe_allow_html=True)
