titanic_desc = """
Build a predictive model to determine which types of passengers were more likely to survive the Titanic disaster, using data such as name, age, gender, and socio-economic class.
"""

titanic_data_info = """
The dataset is divided into a training set and a test set. The training set includes outcomes for each passenger to facilitate model training, while the test set is used to evaluate model performance on unseen data without provided outcomes. Features include gender, class, and others, with opportunities for feature engineering.

Key variables are:

survival: Indicates if a passenger survived (0 = No, 1 = Yes)
pclass: Ticket class as a proxy for socio-economic status (1 = 1st class, 2 = 2nd class, 3 = 3rd class)
sex: Gender of the passenger
age: Age in years (fractional if under 1 year, with .5 indicating an estimated age)
sibsp: Number of siblings or spouses aboard
parch: Number of parents or children aboard
ticket: Ticket number
fare: Fare paid by the passenger
cabin: Cabin number
embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
"""

favorita_grocery_sales_forecasting = """
Develop a machine learning model to forecast product sales for Corporación Favorita, an Ecuadorian grocery retailer. The model should address the complexities of varying store needs, new products, seasonal preferences, and marketing impacts, aiming to optimize inventory and enhance customer satisfaction.
"""


python_and_analyze_data_final_project = """
Is it possible to use the balance information on a bank card to predict the gender of a customer? If so, how accurate is this prediction?
"""

tabular_media_campaign_cost = """
Create a regression model to predict media campaign costs using a tabular dataset, evaluated by root mean squared log error (RMSLE). Predict the target cost for each ID in the test set.
"""

tabular_wild_blueberry_yield = """
Develop a regression model to predict the yield of wild blueberries using a dataset provided. The model's performance will be evaluated based on the Mean Absolute Error (MAE). Predict the target yield for each ID in the test set.
"""

commonlit_evaluate_student_summaries = """
Build a model to evaluate the quality of summaries written by students in grades 3-12, focusing on how well the summary captures the main idea and details of the source text, and the clarity, precision, and fluency of the language. Use a dataset of student summaries for training. Predict scores for two analytic measures for each student_id in the test set. This will aid teachers in assessing student summaries and enable learning platforms to offer immediate feedback.
"""

feedback_prize_2021 = """
Develop a model to segment and classify argumentative and rhetorical elements in essays by students in grades 6-12. Utilize a large dataset of student writings for training in natural language processing. The model's performance will be evaluated based on the overlap between the ground truth and predicted word indices, calculated using Python's .split() function. Submissions must identify strings in the text corresponding to specific classes and provide their word indices. Overlaps of 0.5 or greater for both ground truth to prediction and prediction to ground truth are required for a match, with the best overlap taken in cases of multiple matches. Unmatched predictions and ground truths are considered false positives and false negatives, respectively. For each sample in the test set, submit the sample ID, class, and word indices for each detected string. Multiple entries per class or sample are allowed.
"""