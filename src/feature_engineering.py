import holidays

def create_features(df):

    # Lag features
    df['lag_1'] = df['sales'].shift(1)

    df['lag_7'] = df['sales'].shift(7)

    df['lag_30'] = df['sales'].shift(30)

    # Rolling statistics
    df['rolling_mean'] = (
        df['sales']
        .rolling(7)
        .mean()
    )

    df['rolling_std'] = (
        df['sales']
        .rolling(7)
        .std()
    )

    # Date features
    df['day'] = df['date'].dt.day

    df['month'] = df['date'].dt.month

    df['weekday'] = df['date'].dt.weekday

    # Holiday feature
    india_holidays = holidays.India()

    df['holiday_flag'] = (
        df['date']
        .isin(india_holidays)
        .astype(int)
    )

    # Remove null rows
    df = df.dropna()

    return df