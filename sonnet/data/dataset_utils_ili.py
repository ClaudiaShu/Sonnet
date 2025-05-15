import copy
import numpy as np
import pandas as pd


def plot_val(
    data_y, onset_centre, peak_centre, drop_centre, onset_start, peak_start, drop_start
):
    import matplotlib.pyplot as plt

    # Assuming data_y is a pandas DataFrame or Series with a DatetimeIndex.
    # Convert to Series if needed:
    if isinstance(data_y, pd.DataFrame):
        # For example, if there's a column called "rate" to plot:
        series_to_plot = data_y["rate"]
    else:
        series_to_plot = data_y

    plt.figure(figsize=(12, 6))
    plt.plot(series_to_plot.index, series_to_plot.values, label="data_y", color="blue")

    # Plot vertical dashed lines for onset_centre, peak_centre, and drop_centre.
    # Ensure these are pandas Timestamps.
    for time_point, label, color in zip(
        [onset_centre, peak_centre, drop_centre, onset_start, peak_start, drop_start],
        ["Onset", "Peak", "Drop", "Onset Start", "Peak Start", "Drop Start"],
        ["green", "red", "orange", "blue", "purple", "yellow"],
    ):
        plt.axvline(x=time_point, color=color, linestyle="--", label=label)

    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title("Visualization of data_y with Onset, Peak, and Drop")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig.png")
    plt.close()


def obtain_peak(data):
    assert len(data.values) < 400
    peak_index = np.argmax(data.values)
    peak_date = data.index[peak_index]
    return peak_date


def obtain_onset(data, window=21, threshold=2):
    above_threshold = data > threshold
    onset_index = None
    for i in range(len(data) - window):
        if all(above_threshold[i : i + window]):
            onset_index = i
            break
    if onset_index is not None:
        onset_date = data.index[onset_index]
        return onset_date
    else:
        return None


def obtain_drop(data, window=21, threshold=2):
    above_threshold = data > threshold
    end_index = None
    for i in range(len(data) - 1, window - 1, -1):
        if all(above_threshold[i - window + 1 : i + 1]):
            end_index = i
            break
    if end_index is not None:
        end_date = data.index[end_index]
        return end_date
    else:
        return None


# Function to calculate baseline
def baseline(non_influenza_weeks):
    mean_percentage = non_influenza_weeks.mean()
    std_dev = non_influenza_weeks.std()
    baseline = mean_percentage + 2 * std_dev
    return baseline


def de_trend(t, data):
    slope, intercept = np.polyfit(t, data, 1)  # Fit a linear trend
    # detrended_data = data - (slope * t + intercept)  # Remove the trend
    return slope, intercept


def corr_with(trends, keys):
    # Correlate by a specific period
    trends_ = copy.deepcopy(trends)
    trends_selected = trends_[keys]
    try:
        correlation_score = trends_.corrwith(trends_selected, axis=0, numeric_only=True)
    except Exception:
        correlation_score = trends_.corrwith(trends_selected, axis=0)
    corr_filtered = pd.DataFrame(correlation_score).fillna(0)
    corr_filtered = corr_filtered.reset_index()

    sorted_correlation = correlation_score.sort_values(ascending=False)
    return corr_filtered, sorted_correlation


def get_ili_paths(region):
    if region == "eng":
        return "datasets/ili_trends/eng_ILI.csv"
    elif region == "us2":
        return "datasets/ili_trends/us2_ILI.csv"
    elif region == "us9":
        return "datasets/ili_trends/us9_ILI.csv"
    else:
        raise ValueError(f"Region {region} not recognized.")
