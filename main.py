import os
import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt



def read_data(data_dir):
    try:
        internal_db = pd.read_csv(os.path.join(data_dir, 'CapsLock_internal_db_Test.csv'),
                                  encoding='utf-16', sep='\t')
        leads = pd.read_excel(os.path.join(data_dir, 'CapsLock_Leads_Test.xlsx'))
        tasks = pd.read_excel(os.path.join(data_dir, 'CapsLock_Tasks_Test.xlsx'))
        sets = pd.read_excel(os.path.join(data_dir, 'CapsLock_Sets_Test.xlsx'))
        issued = pd.read_excel(os.path.join(data_dir, 'CapsLock_Issued_Test.xlsx'))

        return internal_db, leads, tasks, sets, issued
    except Exception as e:
        print(f"Error reading files: {e}")
        return None, None, None, None, None


def preprocess_data(internal_db, leads, tasks, sets, issued):

    internal_db['Lavin Media Id'] = internal_db['Lavin Media Id'].astype(int)
    internal_db.rename(columns={
        'Lavin Media Id': 'Lavin Media ID',
        'Created time': 'Lead created time',
        'Thank You page': 'Thank you page'
    }, inplace=True)
    internal_db['Lead created time'] = pd.to_datetime(internal_db['Lead created time'], format='%m/%d/%y %H:%M')

    # Leads preprocessing
    leads['Lavin Media ID'] = leads['Lavin Media ID'].fillna(-1)
    leads['Lavin Media ID'] = leads['Lavin Media ID'].astype(int)
    leads.rename(columns={'Created Time': "Lead created in client's system"}, inplace=True)

    # Tasks preprocessing
    tasks = tasks.dropna(subset=['Lavin Media ID'])
    tasks['Lavin Media ID'] = tasks['Lavin Media ID'].astype(int)
    tasks = tasks.sort_values(['Lavin Media ID', 'Created Time'])

    # Sets preprocessing
    sets.rename(columns={
        'Date': 'Appointment Date',
        'Contact: Contact ID': 'Contact ID',
        'Status': 'Appointment Result',
    }, inplace=True)

    sets['Appointment Date'] = pd.to_datetime(sets['Appointment Date'])
    sets['Appointment Set Date'] = pd.to_datetime(
        sets['Created Date'].astype(str) + ' ' + sets['Created Time'].astype(str))
    sets.drop(columns=['Created Date', 'Created Time'], inplace=True)
    sets = sets.sort_values(['Contact ID', 'Appointment Date'])
    sets = sets.drop_duplicates(subset=['Contact ID'], keep='last')

    # Issued preprocessing
    issued.rename(columns={
        'Contact: Contact ID': 'Contact ID',
        'Gross Sales': 'Gross Sale',
        'Net Sales': 'Net Sale',
        'Cancelled Sales': 'Cancel Sale'
    }, inplace=True)

    issued = issued.sort_values(['Contact ID', 'Date'])
    issued = issued.drop_duplicates(subset=['Contact ID'], keep='last')
    issued = issued[['Contact ID', 'Gross Sale', 'Net Sale', 'Cancel Sale']]

    return internal_db, leads, tasks, sets, issued


def merge_data(internal_db, leads, tasks, sets, issued):
    # Merge internal_db and leads and there were around 145 non-matchings, interestinglt that exist in internabl_db, but not in CRM's of client
    merged_leads_data = pd.merge(internal_db, leads, on='Lavin Media ID', how='inner')

    # Process tasks data
    first_tasks = tasks.groupby('Lavin Media ID').first().reset_index()
    first_tasks = first_tasks[['Lavin Media ID', 'Created Time', 'Call Result']].rename(columns={
        'Created Time': 'First task completed date',
        'Call Result': 'First task Result'
    })

    last_tasks = tasks.groupby('Lavin Media ID').last().reset_index()
    last_tasks = last_tasks[['Lavin Media ID', 'Created Time', 'Call Result']].rename(columns={
        'Created Time': 'Last task Completed date',
        'Call Result': 'Last task Result'
    })

    # Merge with tasks data
    merged_leads_data = merged_leads_data.merge(first_tasks, on='Lavin Media ID', how='left')
    merged_leads_data = merged_leads_data.merge(last_tasks, on='Lavin Media ID', how='left')

    # Merge with sets and issued data
    merged_leads_data = merged_leads_data.merge(sets, on='Contact ID', how='left')
    merged_leads_data = merged_leads_data.merge(issued, on='Contact ID', how='left')

    return merged_leads_data


def clean_leads_data(df):

    df = df.copy()
    df['Contact ID'] = df['Contact ID'].fillna('UNK')
    df['# Call Center Tasks Completed'] = df['# Call Center Tasks Completed'].fillna('0').astype(int)
    df['Net Sale'] = df['Net Sale'].fillna(0).astype(float)
    df['Gross Sale'] = df['Gross Sale'].fillna(0)
    df['Cancel Sale'] = df['Cancel Sale'].astype(float)

    # Case 1 & 2: Completed appointment but Cancel Sale is NaN, so can be set to 0
    df['Cancel Sale'] = np.where(
        (df['Appointment Result'].notna()) & (df['Cancel Sale'].isna()) & (
            ~df['Appointment Result'].isin(["Customer Canceled", "Company Canceled"])),
        0,
        df['Cancel Sale']
    )

    # Case 3: Appointment canceled and Cancel Sale is NaN, so maybe full refund for the future
    df['Cancel Sale'] = np.where(
        (df['Appointment Result'].isin(["Customer Canceled", "Company Canceled"])) & (df['Cancel Sale'].isna()),
        df['Gross Sale'],
        df['Cancel Sale']
    )

    # Case 4: Na replacement with Naive way since if we filter it, we see that they are same NaNs for Sales cols
    df['Cancel Sale'] = df['Cancel Sale'].fillna(0)
    df['Cancel Sale'] = df['Cancel Sale'].astype(int)

    df['Last task Result'] = df['Last task Result'].fillna("Not Attempted")
    df['First task Result'] = df['First task Result'].fillna("Not Attempted")

    df['Appointment Result'] = np.where(
        df['Appointment Date'].isna(),
        "No Appointment",
        df['Appointment Result']
    )

    # Mapping to the same label
    mapping = {'No Answer': 'Not Answered'}
    df['First task Result'] = df['First task Result'].map(mapping).fillna(df['First task Result'])
    df['Last task Result'] = df['Last task Result'].map(mapping).fillna(df['Last task Result'])

    return df


def analyze_data(df):
    # Correlation analysis
    numeric_cols = df.select_dtypes(include='number')
    correlation_matrix = numeric_cols.corr()

    # Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title("Correlation Matrix (Numeric Variables)")
    plt.show()

    return df


def save_results(df, output_dir):

    try:
        output_path = os.path.join(output_dir, 'leads_master_dataset.xlsx')
        df.to_excel(output_path, index=False)
        print(f"Results saved to {output_path}")
    except Exception as e:
        print(f"Error saving results: {e}")


# running my whole pipeline under one func
def main(data_dir, output_dir):

    internal_db, leads, tasks, sets, issued = read_data(data_dir)

    if internal_db is None:
        print("Failed to read data files")
        return

    # Preprocess data
    internal_db, leads, tasks, sets, issued = preprocess_data(internal_db, leads, tasks, sets, issued)

    # Merge data
    merged_leads_data = merge_data(internal_db, leads, tasks, sets, issued)

    # Clean data
    cleaned_data = clean_leads_data(merged_leads_data)

    # Analyze data
    analyzed_data = analyze_data(cleaned_data)

    # Save results
    save_results(analyzed_data, output_dir)

    return analyzed_data



if __name__ == "__main__":
    # My path / output path
    data_directory = "/Users/farizrzayev/Desktop/projects/capslock/_input_data"
    output_directory = "/Users/farizrzayev/Desktop/projects/capslock/output_data"

    # just in case path does not exist
    os.makedirs(output_directory, exist_ok=True)

    final_data = main(data_directory, output_directory)

    print(final_data)