import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

def analyze_sentiment_indico(text):
    url = "https://api.indico.io/sentiment"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {indico_api_key}"
    }
    data = {
        "data": text
    }

    session = requests.Session()
    retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))

    response = session.post(url, json=data, headers=headers)

    if response.status_code == 200:
        result = response.json()
        predicted_sentiment = result["results"]
        return predicted_sentiment
    else:
        print(f"Error: {response.status_code}")
        return None


# Example customer feedback
customer_feedback = "I absolutely love the new features! The customer service is amazing."

# Perform sentiment analysis using Indico API
predicted_sentiment_indico = analyze_sentiment_indico(customer_feedback)

# Display results
print("Customer Feedback:", customer_feedback)
print("Predicted Sentiment (Indico):", predicted_sentiment_indico)



