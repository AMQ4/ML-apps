import pandas as pd
import joblib
import cloudpickle

from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer 
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin

pd.options.mode.copy_on_write = True


X = pd.read_csv("./collection.csv", names=["EnterDate", "DebitID", "number_of_contact", "debit_type", "amount", "collector_id", "collection_amount", "date_of_trans"], header=0)

collection_amount_imp = KNNImputer()
X.collection_amount = collection_amount_imp.fit_transform(X.collection_amount.to_numpy().reshape(-1, 1))

X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

X_train.drop_duplicates(inplace=True)
X_test.drop_duplicates(inplace=True)

y_train = X_train.collection_amount
X_train.drop("collection_amount", axis=1, inplace=True)

y_test = X_test.collection_amount
X_test = X_test.drop("collection_amount", axis=1)


NUMBER_OF_CONTACT_IDX = 0
AMOUNT_IDX = 1
COLLECTION_AMOUNT_IDX = 2
COLLECTORID = 3
DEBIT_TYPE = 4


num_atts = [NUMBER_OF_CONTACT_IDX, AMOUNT_IDX, COLLECTION_AMOUNT_IDX]
cat_atts = [COLLECTORID, DEBIT_TYPE]

rm = ["DebitID", "EnterDate"]

COLUMNS_NAMES = ["number_of_contact",  "amount", "collector_id", "debit_type", "date_of_trans"]


def column_droper(X, idx):
    """
    Drops specified columns from a DataFrame.

    Parameters:
    X : pd.DataFrame
        The input DataFrame from which columns are to be dropped.
    idx : list or str
        The column label(s) to drop.

    Returns:
    pd.DataFrame
        A DataFrame with the specified columns dropped.

    Examples:
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    >>> column_droper(df, 'B')
       A  C
    0  1  7
    1  2  8
    2  3  9

    >>> column_droper(df, ['A', 'C'])
       B
    0  4
    1  5
    2  6
    """
    X_transformed = X.copy()
    # X_transformed.drop_duplicates(inplace=True)
    return X_transformed.drop(idx, axis=1)


class DataFrameTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer for ensuring a DataFrame has specified columns.

    Attributes:
    cols_ : list
        The list of column names that the DataFrame should have.

    Methods:
    fit(X):
        Fits the transformer to the data (no-op).
    transform(X):
        Transforms the input data to a DataFrame with specified columns.

    Examples:
    >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    >>> transformer = DataFrameTransformer(cols=['A', 'B', 'C'])
    >>> transformer.fit(df)
    DataFrameTransformer(cols=['A', 'B', 'C'])
    >>> transformer.transform(df)
       A  B   C
    0  1  3 NaN
    1  2  4 NaN
    """
    
    def __init__(self, cols):
        """
        Initializes the transformer with the specified columns.

        Parameters:
        cols : list
            The list of column names that the DataFrame should have.
        """
        self.cols_ = cols
        pass
    
    def fit(self, X):
        """
        Fits the transformer to the data.

        Parameters:
        X : pd.DataFrame
            The input DataFrame.

        Returns:
        self : DataFrameTransformer
            The fitted transformer.
        """
        return self
    
    def transform(self, X):
        """
        Transforms the input data to a DataFrame with specified columns.

        Parameters:
        X : pd.DataFrame
            The input DataFrame.

        Returns:
        pd.DataFrame
            A DataFrame with the specified columns.
        """
        return pd.DataFrame(X, columns=self.cols_)

class TotalCollectorsTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that adds a column indicating the total number of collectors for each transaction date.

    Attributes:
    collectors_map_ : dict
        A dictionary mapping each transaction date to the total number of collectors for that date.

    Methods:
    fit(X):
        Fits the transformer to the data by calculating the total number of collectors for each transaction date.
    transform(X):
        Transforms the input data by adding a column with the total number of collectors for each transaction date.

    Examples:
    >>> df = pd.DataFrame({
    ...     'date_of_trans': ['2023-01-01', '2023-01-01', '2023-01-02'],
    ...     'collector_id': [1, 2, 3]
    ... })
    >>> transformer = TotalCollectorsTransformer()
    >>> transformer.fit(df)
    TotalCollectorsTransformer()
    >>> transformer.transform(df)
      date_of_trans  collector_id  total_number_of_collectors
    0    2023-01-01             1                           2
    1    2023-01-01             2                           2
    2    2023-01-02             3                           1
    """
    
    def __init__(self):
        """
        Initializes the transformer.

        Attributes:
        collectors_map_ : dict
            Will hold the mapping of transaction dates to the total number of collectors.
        """
        self.collectors_map_ = None
    
    def fit(self, X):
        """
        Fits the transformer to the data by calculating the total number of collectors for each transaction date.

        Parameters:
        X : pd.DataFrame
            The input DataFrame with 'date_of_trans' and 'collector_id' columns.

        Returns:
        self : TotalCollectorsTransformer
            The fitted transformer.
        """
        X_grouped_by_date = X.groupby(by="date_of_trans")
        X_agged = X_grouped_by_date["collector_id"].count().reset_index(name="total_number_of_collectors")
        self.collectors_map_ = dict(zip(tuple(X_agged.date_of_trans), tuple(X_agged.total_number_of_collectors)))
        
        return self
    
    def transform(self, X):
        """
        Transforms the input data by adding a column with the total number of collectors for each transaction date.

        Parameters:
        X : pd.DataFrame
            The input DataFrame with 'date_of_trans' and 'collector_id' columns.

        Returns:
        pd.DataFrame
            The transformed DataFrame with an additional 'total_number_of_collectors' column.

        Raises:
        NotFittedError:
            If the transformer is used before calling 'fit'.
        """
        if self.collectors_map_ is None:
            raise NotFittedError(f"This {self.__class__.__name__} instance is not fitted yet. "
                                 f"Call 'fit' with appropriate arguments before using this estimator.")
        X_transformed = X.copy()
        X_transformed["total_number_of_collectors"] = X_transformed["date_of_trans"].map(self.collectors_map_)
        
        return X_transformed
    
        
class DebitTypeContributionTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that adds a column indicating the contribution of each debit type to the collected amount.

    Attributes:
    contribution_map_ : dict
        A dictionary mapping each debit type to its calculated contribution.
    global_cont_mean_ : float
        The global mean contribution, used to fill missing contributions.

    Methods:
    fit(X):
        Fits the transformer to the data by calculating the contribution of each debit type.
    transform(X):
        Transforms the input data by adding a column with the contribution of each debit type.

    Examples:
    >>> df = pd.DataFrame({
    ...     'debit_type': ['A', 'A', 'B', 'C'],
    ...     'amount': [100, 150, 200, 250],
    ...     'collection_amount': [50, 75, 150, 125]
    ... })
    >>> transformer = DebitTypeContributionTransformer()
    >>> transformer.fit(df)
    DebitTypeContributionTransformer()
    >>> transformer.transform(df)
       amount  collection_amount  debit_type_contribution
    0     100                 50                1.00
    1     150                 75                1.00
    2     200                150                0.75
    3     250                125                0.50
    """
    
    def __init__(self):
        """
        Initializes the transformer.

        Attributes:
        contribution_map_ : dict
            Will hold the mapping of debit types to their contributions.
        global_cont_mean_ : float
            Will hold the global mean contribution for debit types.
        """
        self.contribution_map_ = None
        self.global_cont_mean_ = None

    def fit(self, X):
        """
        Fits the transformer to the data by calculating the contribution of each debit type.

        Parameters:
        X : pd.DataFrame
            The input DataFrame with 'debit_type', 'amount', and 'collection_amount' columns.

        Returns:
        self : DebitTypeContributionTransformer
            The fitted transformer.
        """
        X_grouped = X.groupby(by="debit_type")
        
        debit_type_counts = X_grouped["debit_type"].count().reset_index(name="counts")
        amount_sum = X_grouped["amount"].sum().reset_index(name="amount_sum").amount_sum
        collection_amount_sum = X_grouped["collection_amount"].sum().reset_index(name="collection_amount_sum").collection_amount_sum
        
        contribution = debit_type_counts.counts * collection_amount_sum / amount_sum
        self.global_cont_mean_ = contribution.mean()
        self.contribution_map_ = dict(zip(tuple(debit_type_counts.debit_type), tuple(contribution)))
        
        return self

    def transform(self, X):
        """
        Transforms the input data by adding a column with the contribution of each debit type.

        Parameters:
        X : pd.DataFrame
            The input DataFrame with 'debit_type' column.

        Returns:
        pd.DataFrame
            The transformed DataFrame with an additional 'debit_type_contribution' column and without 'debit_type' column.

        Raises:
        NotFittedError:
            If the transformer is used before calling 'fit'.
        """
        if self.contribution_map_ is None:
            raise NotFittedError(f"This {self.__class__.__name__} instance is not fitted yet. "
                                 f"Call 'fit' with appropriate arguments before using this estimator.")
        
        X_transformed = X.copy()
        
        X_transformed["debit_type_contribution"] = X_transformed["debit_type"].map(self.contribution_map_)
        X_transformed["debit_type_contribution"] = X_transformed["debit_type_contribution"].fillna(self.global_cont_mean_)
                
        return X_transformed.drop("debit_type", axis=1)
    
    
class CollectorPerformanceTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that adds columns indicating the daily and overall performance of collectors.

    Attributes:
    daily_coll_contribution_map_ : dict
        A dictionary mapping each collector ID to their daily performance contribution.
    overall_coll_contribution_map_ : dict
        A dictionary mapping each collector ID to their overall performance contribution.
    daily_cont_global_mean_ : float
        The global mean of daily performance contributions.
    overall_cont_global_mean_ : float
        The global mean of overall performance contributions.

    Methods:
    fit(X):
        Fits the transformer to the data by calculating the daily and overall performance of each collector.
    transform(X):
        Transforms the input data by adding columns with the daily and overall performance of each collector.

    Examples:
    >>> df = pd.DataFrame({
    ...     'date_of_trans': ['2023-01-01', '2023-01-01', '2023-01-02'],
    ...     'collector_id': [1, 2, 1],
    ...     'amount': [100, 200, 150],
    ...     'collection_amount': [80, 160, 120]
    ... })
    >>> transformer = CollectorPerformanceTransformer()
    >>> transformer.fit(df)
    CollectorPerformanceTransformer()
    >>> transformer.transform(df)
       collector_id  amount  collection_amount  daily_collector_contribution  overall_collector_contribution
    0             1     100                 80                       0.800000                        0.800000
    1             2     200                160                       0.800000                        0.800000
    2             1     150                120                       0.800000                        0.800000
    """
    
    def __init__(self):
        """
        Initializes the transformer.

        Attributes:
        daily_coll_contribution_map_ : dict
            Will hold the mapping of collector IDs to their daily performance contributions.
        overall_coll_contribution_map_ : dict
            Will hold the mapping of collector IDs to their overall performance contributions.
        daily_cont_global_mean_ : float
            Will hold the global mean of daily performance contributions.
        overall_cont_global_mean_ : float
            Will hold the global mean of overall performance contributions.
        """
        self.daily_coll_contribution_map_ = None
        self.overall_coll_contribution_map_ = None
        self.daily_cont_global_mean_ = None
        self.overall_cont_global_mean_ = None

    def fit(self, X):
        """
        Fits the transformer to the data by calculating the daily and overall performance of each collector.

        Parameters:
        X : pd.DataFrame
            The input DataFrame with 'date_of_trans', 'collector_id', 'amount', and 'collection_amount' columns.

        Returns:
        self : CollectorPerformanceTransformer
            The fitted transformer.
        """
        X_grouped_by_date_collid = X.groupby(by=["date_of_trans", "collector_id"])
        X_agged_date_collid = X_grouped_by_date_collid[["collection_amount", "amount"]].sum().reset_index()
        
        X_grouped_collid = X.groupby(by=["collector_id"])
        X_agged_collid = X_grouped_collid[["collection_amount", "amount"]].sum().reset_index()
        
        daily_collector_performance = X_agged_date_collid.collection_amount / X_agged_date_collid.amount        
        overall_collector_performance = X_agged_collid.collection_amount / X_agged_collid.amount        
        
        self.daily_coll_contribution_map_ = dict(zip(tuple(X_agged_date_collid.collector_id), tuple(daily_collector_performance)))
        self.overall_coll_contribution_map_ = dict(zip(tuple(X_agged_collid.collector_id), tuple(overall_collector_performance)))
        self.daily_cont_global_mean_ = daily_collector_performance.mean()
        self.overall_cont_global_mean_ = overall_collector_performance.mean()
        
        return self

    def transform(self, X):
        """
        Transforms the input data by adding columns with the daily and overall performance of each collector.

        Parameters:
        X : pd.DataFrame
            The input DataFrame with 'collector_id' and 'date_of_trans' columns.

        Returns:
        pd.DataFrame
            The transformed DataFrame with additional 'daily_collector_contribution' and 'overall_collector_contribution' columns and without 'date_of_trans' column.

        Raises:
        NotFittedError:
            If the transformer is used before calling 'fit'.
        """
        if self.daily_coll_contribution_map_ is None or self.overall_coll_contribution_map_ is None:
            raise NotFittedError(f"This {self.__class__.__name__} instance is not fitted yet. "
                                 f"Call 'fit' with appropriate arguments before using this estimator.")
        
        X_transformed = X.copy()
        
        X_transformed["daily_collector_contribution"] = X_transformed["collector_id"].map(self.daily_coll_contribution_map_)
        X_transformed["overall_collector_contribution"] = X_transformed["collector_id"].map(self.overall_coll_contribution_map_)
        
        X_transformed["daily_collector_contribution"] = X_transformed["daily_collector_contribution"].fillna(self.daily_cont_global_mean_)
        X_transformed["overall_collector_contribution"] = X_transformed["overall_collector_contribution"].fillna(self.overall_cont_global_mean_)

        return X_transformed.drop(["date_of_trans"], axis=1)

    
class FittedTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that applies a given fitted transformer to the data without refitting it.

    Attributes:
    transformer : Transformer
        A pre-fitted transformer that will be used to transform the data.

    Methods:
    fit(X):
        A no-op method that returns the instance itself. This method does not refit the transformer.
    transform(X):
        Transforms the input data using the provided pre-fitted transformer.

    Examples:
    >>> from sklearn.preprocessing import StandardScaler
    >>> import pandas as pd
    >>> data = pd.DataFrame({'feature': [1, 2, 3, 4, 5]})
    >>> scaler = StandardScaler().fit(data)
    >>> fitted_transformer = FittedTransformer(transformer=scaler)
    >>> fitted_transformer.fit(data)
    FittedTransformer(transformer=StandardScaler())
    >>> transformed_data = fitted_transformer.transform(data)
    >>> print(transformed_data)
       feature
    0 -1.414214
    1 -0.707107
    2  0.000000
    3  0.707107
    4  1.414214
    """
    
    def __init__(self, transformer):
        """
        Initializes the FittedTransformer with a pre-fitted transformer.

        Parameters:
        transformer : Transformer
            A pre-fitted transformer that will be used to transform the data.
        """
        self.transformer = transformer

    def fit(self, X):
        """
        A no-op method that returns the instance itself. This method does not refit the transformer.

        Parameters:
        X : pd.DataFrame or np.ndarray
            The input data.

        Returns:
        self : FittedTransformer
            The instance itself.
        """
        return self

    def transform(self, X):
        """
        Transforms the input data using the provided pre-fitted transformer.

        Parameters:
        X : pd.DataFrame or np.ndarray
            The input data to be transformed.

        Returns:
        pd.DataFrame or np.ndarray
            The transformed data.
        """
        return self.transformer.transform(X)

dropper = FunctionTransformer(column_droper, kw_args={"idx": rm})

imp_pipeline = ColumnTransformer([
        ("num_imp", KNNImputer(), ["number_of_contact", "amount"]),
        ("cat_imp", SimpleImputer(strategy="most_frequent"), ["collector_id", "debit_type"])
    ], remainder="passthrough"
)

coll_per_trans = joblib.load("./fitted_coll_per_trans.pkl")
debit_type_trans = joblib.load("./fitted_debit_type_trans.pkl")

full_pipeline = Pipeline([
    ("dropper", dropper),
    ("imp_pipeline", imp_pipeline),
    ("df_trans", DataFrameTransformer(cols=COLUMNS_NAMES)),
    #("total_collectors_adder", TotalCollectorsTransformer()),
    ("coll_per_trans", FittedTransformer(coll_per_trans)),
    ("debit_type_trans", FittedTransformer(debit_type_trans)),
    ("scaller", StandardScaler())
])


dropper = FunctionTransformer(column_droper, kw_args={"idx": rm})

imp_pipeline = ColumnTransformer([
        ("num_imp", KNNImputer(), ["number_of_contact", "amount"]),
        ("cat_imp", SimpleImputer(strategy="most_frequent"), ["collector_id", "debit_type"])
    ], remainder="passthrough"
)

coll_per_trans = joblib.load("./fitted_coll_per_trans.pkl")
debit_type_trans = joblib.load("./fitted_debit_type_trans.pkl")

full_pipeline = Pipeline([
    ("dropper", dropper),
    ("imp_pipeline", imp_pipeline),
    ("df_trans", DataFrameTransformer(cols=COLUMNS_NAMES)),
    #("total_collectors_adder", TotalCollectorsTransformer()),
    ("coll_per_trans", FittedTransformer(coll_per_trans)),
    ("debit_type_trans", FittedTransformer(debit_type_trans)),
    ("scaller", StandardScaler())
])


full_pipeline.fit_transform(X_train)

with open('preprocessing_pipeline.pkl', 'wb') as file:
    cloudpickle.dump(full_pipeline, file)