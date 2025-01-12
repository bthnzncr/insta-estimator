# insta-estimator

This repository contains two predictive tasks:
1. A regression task that predicts the like count on social media posts.
2. A category classification task that leverages text and profile information.

In this README, I explain the rationale behind my feature selection, preprocessing, and model choices. The document is divided into two main sections: **Regression** and **Category Classification**.

---

## Part 1: Regression Task – Predicting Like Counts

### Feature Selection and Engineering

In the regression pipeline, objective is to predict the number of likes for an individual post. I selected features that capture both **user-level** and **post-level** characteristics to help explain post performance.

#### Key Features

- **user_avg_likes**  
  *Rationale:* I use the average likes per user as a baseline, setting expectations for what a typical post might receive.

- **user_avg_comments**  
  *Rationale:* Comments indicate engagement. Comparing a post's comment count with the user’s average helps signal when engagement deviates from the norm.

- **comment_count**  
  *Rationale:* This directly correlates with how engaging a post is. Posts receiving many comments are likely to have high like counts as well.
  
- **norm_comment_ratio**  
  *Rationale:* By normalizing the comment count relative to the user’s overall engagement, this feature assesses whether the current post is under- or over-performing compared to historical trends.
  *it is worth to mention that I tried to implement some pre-prediction like count = average_like * (comment / average_comment) with some average_like_count like over 1000, but it did not perform well...
- **following_count** & **follower_count**  
  *Rationale:* These metrics reflect the user’s network size and potential reach. A larger follower base often drives higher engagement.

- **num_hashtags**  
  *Rationale:* Hashtags can amplify a post’s visibility, which in turn may lead to more likes especially small accounts.

- **is_business** and **is_verified**  
  *Rationale:* Verified and business profiles may receive different engagement patterns due to brand recognition and trust.

- **post_dayofweek**  
  *Rationale:* Posting time matters. The day of the week can affect how many users see and interact with a post, expecially for the weekends.

*Additional derived features (e.g., text_length, num_mentions, num_emojis) were computed during feature extraction. However, my final regression model uses a focused set of features that directly influence the like counts.*

### Model Choice: RandomForestRegressor

I selected the **RandomForestRegressor** for several reasons:
- **Handling Non-linearity:** It effectively captures the non-linear interactions among features.
- **Robustness:** The ensemble approach mitigates overfitting and handles outliers in noisy social media data.
- **Feature Importance:** It offers insights into which features contribute most to prediction performance.
- **Ease of Implementation:** Hyperparameter tuning (e.g., `n_estimators`, `max_depth`) has shown robust performance on my dataset.

The regression pipeline involves:
1. Loading and preprocessing the dataset.
2. Extracting features at both the user and post levels.
3. Normalizing selected numerical features.
4. Splitting the data into training and a fixed-size validation set.
5. Training the RandomForestRegressor and evaluating predictions.
6. Generating predictions for the official test set with consistent feature extraction.

---

## Part 2: Category Classification Task

The second pipeline—targeting category classification—is designed to predict a user’s category label (e.g., based on biography, content themes, etc.). The additional code segments introduce rich text and profile-based feature channels along with advanced preprocessing.

### Key Additions in This Pipeline

#### 1. Advanced Text Preprocessing


- **Preprocessing Functions:**
  - **`preprocess_text(text)`**  
    *Functionality:* Lowercases the text, removes URLs, non-alphanumeric characters (excluding `#` and `@`), digits, and extra whitespace, and filters out any tokens present in `WORDS_TO_EXCLUDE`.
  
  - **`quick_extract_emojis(text)`**  
    *Functionality:* Uses a simple regex to extract emojis from text. Emojis are handled separately to retain their contextual meaning.
  
  - **`extract_tags(original_text)`**  
    *Functionality:* Extracts hashtags and mentions from the text, preprocesses them, and filters out excluded tokens.

These functions ensure that my text data is clean and standardized before vectorization.

#### 2. Building Multiple Textual and Numeric Feature Channels

For each user, I extract and process different channels of information:

- **Captions + Biography Channel (`capbio_corpus`):**  
  Merges all preprocessed captions and biography text.

- **Full Name Channel (`fullname_corpus`):**  
  Processes the full names provided in the profile.

- **Tags Channel (`tags_corpus`):**  
  Aggregates hashtags and mentions from both posts and the biography.

- **Category Name Channel (`cat_name_corpus`):**  
  Uses the profile’s category name after cleaning.

- **Emoji Channel (`emoji_corpus`):**  
  Collects emojis separately, ensuring that the emotional or visual content is preserved.

- **Average Post Hour (`avg_post_hour_list`):**  
  A newly added numerical feature computed as the average posting hour per user, which can capture temporal trends in engagement.

#### 3. Vectorization & Feature Combination

Each textual channel is vectorized using **TF-IDF Vectorization**:
- Different maximum feature sizes and stop word settings are used for each channel to best capture the information from that text.
  
Boolean features and ratio features are also extracted, including:
- **Profile Booleans:** (is_business, is_supervision_enabled, is_verified, is_professional_account)
- **Follower-to-Following Ratio:** Normalized to capture user influence.
- **Average Post Hour:** Normalized to a [0, 1] range.

All channels are **horizontally combined** (using `hstack`) to form a consolidated feature matrix for each user.

#### 4. Dimensionality Reduction & Class Balancing

- **SMOTE:**  
  Applied after scaling the (sparse) features to balance class distribution. This ensures that minority classes are sufficiently represented during training.

- **TruncatedSVD:**  
  Utilized to reduce the dimensionality of the large combined feature space while preserving as much information as possible. This step reduces computational load and helps with generalization.

#### 5. Classification Model: Logistic Regression

- **Model Selection:**  
  I chose **Logistic Regression** for its interpretability and efficiency. It is trained on the SVD-reduced features.

- **Training & Evaluation:**  
  After splitting the data into training and validation sets (with a 90:10 ratio), I apply scaling and SMOTE followed by dimensionality reduction. The Logistic Regression model is then trained, and its performance is evaluated using accuracy and detailed classification reports.

#### 6. Prediction on Official Test Users

For the official test set:
- I process each test user’s profile and posts using the same channels and preprocessing pipelines.
- I then vectorize, combine, and scale the test features.
- Finally, the trained Logistic Regression model predicts the category for each test user and saves the output as a JSON file.

---


