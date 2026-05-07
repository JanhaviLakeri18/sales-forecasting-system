import pandas as pd


def load_data(filepath):

    # ==========================
    # LOAD CSV
    # ==========================

    df = pd.read_csv(filepath)

    # ==========================
    # REMOVE EXTRA SPACES
    # ==========================

    df.columns = df.columns.str.strip()

    # ==========================
    # RENAME COLUMNS
    # ==========================

    df = df.rename(columns={

        'State': 'state',

        'Date': 'date',

        'Total': 'sales',

        'Category': 'category'
    })

    # ==========================
    # DATE CONVERSION
    # ==========================

    df['date'] = pd.to_datetime(

        df['date'],

        format='mixed',

        dayfirst=True
    )

    # ==========================
    # CLEAN SALES COLUMN
    # ==========================

    df['sales'] = (

        df['sales']

        .astype(str)

        .str.replace(',', '')

        .astype(float)
    )

    # ==========================
    # SORT DATA
    # ==========================

    df = df.sort_values('date')

    # ==========================
    # RESET INDEX
    # ==========================

    df = df.reset_index(drop=True)

    return df