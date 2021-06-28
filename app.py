# save this as app.py
from flask import Flask, escape, request ,render_template

import pickle
model = pickle.load(open('GB_Model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/analysis')
def analysis():
    return render_template("churn.html")

@app.route('/prediction', methods=['GET' , 'POST'])
def prediction():

    if request.method == "POST" :

        age = request.form['age']
        last_login = int(request.form['last_login'])
        avg_time_spent = float(request.form['avg_time_spent'])
        avg_transaction_value = float(request.form['avg_transaction_value'])
        points_in_wallet = float(request.form['points_in_wallet'])
        date = request.form['date']
        avg_frequency_login_days = float(request.form['avg_frequency_login_days'])
        gender = request.form['gender']
        region_category = request.form['region_category']
        membership_category = request.form['membership_category']
        joined_through_referral = request.form['joined_through_referral']
        preferred_offer_types = request.form['preferred_offer_types']
        medium_of_operation = request.form['medium_of_operation']
        internet_option = request.form['internet_option']
        used_special_discount = request.form['used_special_discount']
        offer_application_preference = request.form['offer_application_preference']
        past_complaint = request.form['past_complaint']
        complaint_status = request.form['complaint_status']
        feedback = request.form['feedback']


        # gender
        if gender == "M":
            gender=1
        else:
            gender=0

        # region_category
        if region_category == 'Town':
            region_category=1
        elif region_category == 'Village':
            region_category=2
        else:
            region_category=0
        
        # membership_category
        if membership_category == 'Basic Membership':
            membership_category=0
        elif membership_category == 'No Membership':
            membership_category=2
        elif membership_category == 'Gold Membership':
            membership_category=1
        elif membership_category == 'Platinum Membership':
            membership_category=3
        elif membership_category == 'Silver Membership':
            membership_category=5
        else:
            membership_category=4
            
        # joined_through_referral
        if joined_through_referral == 'Yes':
            joined_through_referral=1
        else:
            joined_through_referral=0

        # preferred_offer_types
        if preferred_offer_types == 'Gift Vouchers/Coupons':
            preferred_offer_types=1
        elif preferred_offer_types == 'Credit/Debit Card Offers':
            preferred_offer_types=0
        else:
            preferred_offer_types=2

        # medium_of_operation
        if medium_of_operation == 'Desktop':
            medium_of_operation=1
        elif medium_of_operation == 'Smartphone':
            medium_of_operation=2
        else:
            medium_of_operation=0

        # internet_option
        if internet_option == 'Wi-Fi':
            internet_option=2
        elif internet_option == 'Mobile_Data':
            internet_option=1
        else:
            internet_option=0

            
        # used_special_discount
        if used_special_discount == 'Yes':
            used_special_discount=1
        else:
            used_special_discount=0

        # offer_application_preference
        if offer_application_preference == 'Yes':
            offer_application_preference=1
        else:
            offer_application_preference=0

        # past_complaint
        if past_complaint == 'Yes':
            past_complaint=1
        else:
            past_complaint=0

        # complaint_status
        if complaint_status == 'Not Applicable':
            complaint_status=1
        elif complaint_status == 'Solved in Follow-up':
            complaint_status=3
        elif complaint_status == 'No Information Available':
            complaint_status=0
        elif complaint_status == 'Unsolved':
            complaint_status=4
        else:
            complaint_status=2

        # feedback
        if feedback == 'Poor Product Quality':
            feedback=2
        elif feedback == 'No reason specified':
            feedback=0
        elif feedback == 'Too many ads':
            feedback=7
        elif feedback == 'Poor Website':
            feedback=3
        elif feedback == 'Poor Customer Service':
            feedback=1 
        elif feedback == 'Reasonable Price':
            feedback=6
        elif feedback == 'User Friendly Website':
            feedback=8
        elif feedback == 'Products always in Stock':
            feedback=4
        else:
            feedback=5



        date2 = date.split('-')
        joining_day = int(date2[0])
        joining_month = int(date2[1])
        joining_year = int(date2[2])


        data = { 
            'age':[age],
            'gender':[gender],
            'region_category':[region_category],
            'membership_category':[membership_category],
            'joined_through_referral':[joined_through_referral],
            'preferred_offer_types':[preferred_offer_types],
            'medium_of_operation':[medium_of_operation],
            'internet_option':[internet_option],
            'days_since_last_login':[last_login],
            'avg_time_spent':[avg_time_spent],
            'avg_transaction_value':[avg_transaction_value],
            'avg_frequency_login_days':[avg_frequency_login_days],
            'points_in_wallet':[points_in_wallet],
            'used_special_discount':[used_special_discount],
            'offer_application_preference':[offer_application_preference],
            'past_complaint':[past_complaint],
            'complaint_status':[complaint_status],
            'feedback':[feedback],
            'joining_day':[joining_day],
            'joining_month':[joining_month],
            'joining_year':[joining_year]
            }

        import pandas as pd

        df = pd.DataFrame.from_dict(data)

        pred = model.predict(df)
        #print(pred)


        return render_template("prediction.html" , pred_text="Churn Score is {}".format(pred))

    else :
        return render_template("prediction.html")

if __name__ == "__main__":
    app.run(debug=True)

