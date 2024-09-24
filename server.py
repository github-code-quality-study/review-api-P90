import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

reviews = pd.read_csv('data/reviews.csv').to_dict('records')

ALLOWED_LOCATIONS = ["Albuquerque, New Mexico",
"Carlsbad, California"
"Chula Vista, California"
"Colorado Springs, Colorado"
"Denver, Colorado"
"El Cajon, California",
"El Paso, Texas",
"Escondido, California",
"Fresno, California",
"La Mesa, California",
"Las Vegas, Nevada",
"Los Angeles, California",
"Oceanside, California",
"Phoenix, Arizona",
"Sacramento, California",
"Salt Lake City, Utah",
"Salt Lake City, Utah",
"San Diego, California",
"Tucson, Arizona"]

TIMESTAMP_FORMAT = '%Y-%m-%d %H:%M:%S'

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        # This method is a placeholder for future initialization logic
        pass

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        """
        The environ parameter is a dictionary containing some useful
        HTTP request information such as: REQUEST_METHOD, CONTENT_LENGTH, QUERY_STRING,
        PATH_INFO, CONTENT_TYPE, etc.
        """

        if environ["REQUEST_METHOD"] == "GET":
            # Extract query parameters
            params = parse_qs(environ["QUERY_STRING"])
            input_location = params.get("location")[0] if params.get("location") else None
            input_start_date = params.get("start_date")[0] if params.get("start_date") else None
            input_end_date = params.get("end_date")[0] if params.get("end_date") else None

            start_date = None
            end_date = None
            # Convert query dates to datetime format
            if input_start_date:
                start_date = datetime.strptime(input_start_date, '%Y-%m-%d')
            if input_end_date:
                end_date = datetime.strptime(input_end_date, '%Y-%m-%d')

            reviews_with_sentiments = []
            for review in reviews:
                review_date = datetime.strptime(review['Timestamp'], TIMESTAMP_FORMAT)

                add_review = False
                if start_date and end_date:
                    if start_date <= review_date <=  end_date:
                        add_review = True
                elif start_date:
                    if review_date >= start_date:
                        add_review = True      
                elif end_date:
                    if review_date <= end_date:
                        add_review = True
                elif input_location:
                    if review["Location"] == input_location:
                        add_review = True
                else:
                    add_review = True 

                # Add review with sentiment
                if add_review:
                    review.update({"sentiment" : self.analyze_sentiment(review["ReviewBody"])})
                    reviews_with_sentiments.append(review)
    
            sorted_by_sentiment = sorted(reviews_with_sentiments, key=lambda x: x['sentiment']['compound'], reverse=True)
            response_body = json.dumps(sorted_by_sentiment, indent=2).encode("utf-8")
            

            # Send response
            start_response("200 OK", [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(response_body)))
             ])
            
            return [response_body]


        if environ["REQUEST_METHOD"] == "POST":
            # Extract post data
            content_length = int(environ["CONTENT_LENGTH"])
            data = parse_qs(environ['wsgi.input'].read(content_length).decode("utf-8"))

            # Extract data parameters
            location = data.get("Location")[0] if data.get("Location") else None
            review_body = data.get("ReviewBody")[0] if data.get("ReviewBody") else None

            review = {}
            if review_body and location and location in ALLOWED_LOCATIONS:
                status = "201 OK"
                review = {"ReviewBody" : review_body,
                 "Location" : location,
                 "ReviewId" : str(uuid.uuid4()),
                 "Timestamp" : str(datetime.now().replace(microsecond=0))}
            else:
                status = "400 Bad Request"
                response_body = json.dumps(review, indent=2).encode("utf-8")

            # Send response
            response_body = json.dumps(review, indent=2).encode("utf-8")
            start_response(status, [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body)))
            ])
            return [response_body]


if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get('PORT', 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()