import streamlit as st

def home():
    st.write("Classification Project")
    st.title("Telco Churn Classification App")
    st.markdown("""
    This app uses machine learning to classify whether a customer is likely
    to churn or not""")
    st.subheader("Instructions")
    st.markdown("""
    - Upload a csv file
    - Select the features for classification
    - Choose a machine learning model from the dropdown
    - Click on 'Classify' to get the predicted results.
    - The app gives you a report on the performance of the model
    - Expect it to give metrics like f1 score, recall, precision and accura
    """)

    st.header("App Features")
    st.markdown("""
    -  Data View: Access the customer data.
    -  Dashboard: Shows data visualizations for insights
    """)

    st.subheader("User Benefits")
    st.markdown("""
    - **Data Driven Decisions: You make an informed decision backed by da
    - **Access Machine Learning: Utilize machine learning algorithms.
    """)
    st.write("#### How to Run the application") 
    with st.container (border=True):
        st.code(""" #Activate the virtual environment 
        env/scripts/activate
    
    #Run the App
    streamlit run p.py
            """)


#adding the embeded link

    st.video("https://www.youtube.com/watch?v=fMM54UG4a8A&list=PPSV", autoplay=True)

    #adding the clickable link

    st.markdown("""[Watch a Demo](https://www.youtube.com/watch?v=fMM54UG4a8A&list=PPSV)
                """)

if __name__ == "__main__":
    home()