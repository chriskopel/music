### This has been modified from its original to match 2024_12_27__attempt_2

import os
import subprocess
from datetime import date

import librosa
import numpy as np
import pandas as pd
import json
import glob

import youtube_dl

import gzip
import pyarrow as pa
import pyarrow.parquet as pq

import warnings
warnings.filterwarnings("ignore") # Suppress all warnings

import time


### Librosa functions
def extract_audio_data(audio_fp):
    y, sr = librosa.load(audio_fp)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    chromagram = librosa.feature.chroma_stft(y=y,sr=sr)
    tonnetz = librosa.feature.tonnetz(y=y,sr=sr)
    return mfcc, melspectrogram, chromagram, tonnetz



def audio_data_summary_stats(audio_fp):
    # Get audio librosa features
    mfcc, melspectrogram, chromagram, tonnetz = extract_audio_data(audio_fp)

    # MFCC
    mfcc_summary_stats = np.concatenate((
        np.min(mfcc, axis=1),
        np.percentile(mfcc, 25, axis=1),
        np.mean(mfcc, axis=1),
        np.median(mfcc, axis=1),
        np.percentile(mfcc, 75, axis=1),
        np.max(mfcc, axis=1)
    ))

    # Mel-spec
    mel_spec_summary_stats = np.concatenate((
        np.min(melspectrogram, axis=1),
        np.percentile(melspectrogram, 25, axis=1),
        np.mean(melspectrogram, axis=1),
        np.median(melspectrogram, axis=1),
        np.percentile(melspectrogram, 75, axis=1),
        np.max(melspectrogram, axis=1)
    ))

    # Chromagram
    chromagram_summary_stats = np.concatenate((
        np.min(chromagram, axis=1),
        np.percentile(chromagram, 25, axis=1),
        np.mean(chromagram, axis=1),
        np.median(chromagram, axis=1),
        np.percentile(chromagram, 75, axis=1),
        np.max(chromagram, axis=1)
    ))

    # Tonnetz 
    tonnetz_summary_stats = np.concatenate((
        np.min(tonnetz, axis=1),
        np.percentile(tonnetz, 25, axis=1),
        np.mean(tonnetz, axis=1),
        np.median(tonnetz, axis=1),
        np.percentile(tonnetz, 75, axis=1),
        np.max(tonnetz, axis=1)
    ))

    # Combine all summary stats
    all_summary_stats = np.concatenate((
        mfcc_summary_stats, mel_spec_summary_stats, chromagram_summary_stats, tonnetz_summary_stats
    ))

    return all_summary_stats





### YT specific functions
def yt_dl_pl_bulk(playlist_url, yt_audio_out_fp, yt_metadata_out_fp, audio_file_type):
	### YT DL (audio then metadata)
	## Audio DL:
	# Set the options for downloading the audio files
	audio_options = {
		'format': 'bestaudio/best',
		'postprocessors': [{
			'key': 'FFmpegExtractAudio',
			'preferredcodec': audio_file_type,# 'flac', # 'mp3'
			'preferredquality': '0' if audio_file_type == 'flac' else '192' # '0' #  '192'
		}],
		'outtmpl': f'{yt_audio_out_fp}/%(title)s.%(ext)s',
		'updatetime': False,  # Do not update the output file if it already exists
		'quiet': True # supress output
	}
	
	# Create a YouTube downloader object for downloading audio files
	audio_downloader = youtube_dl.YoutubeDL(audio_options)
	
	# Download the playlist and save the audio files
	audio_downloader.download([playlist_url])

	# Grab num files
	audio_out_file_list = os.listdir(yt_audio_out_fp)
	num_audio_out_files = len(audio_out_file_list)
	
	
	## Metadata DL:
	# Set the options for downloading the JSON metadata files
	json_options = {
		'writeinfojson': True,  # Save JSON metadata for each video
		'skip_download': True, # don't dl the video file
		'writedescription': False,  # Do not save video description
		'updatetime': False,  # Do not update the output file if it already exists
		'outtmpl': f'{yt_metadata_out_fp}/%(title)s.json',
		'quiet': True
	}
	
	# Create a YouTube downloader object for downloading JSON metadata files
	json_downloader = youtube_dl.YoutubeDL(json_options)
	
	# Download the playlist and save the JSON metadata files
	json_downloader.download([playlist_url])

	# Grab num files
	json_out_file_list = os.listdir(yt_metadata_out_fp)
	num_json_out_files = len(json_out_file_list)

	return_msg = f"{num_audio_out_files} audio files have been placed in {yt_audio_out_fp} and {num_json_out_files} json files have been placed in {yt_metadata_out_fp}"

	print(return_msg)
	
	
### Deleted yt_dl_folder_prep
	
	


def yt_audio_metadata_extraction(yt_audio_out_fp, audio_data_fp):
	### Step 3: MFCC extraction
	audio_file_list = os.listdir(yt_audio_out_fp)
	audio_data_fp_list = []
	
	for i, audio_file in enumerate(audio_file_list):
		## Music data
		# Extract audio data summary stats
		summary_stats_feats = audio_data_summary_stats(os.path.join(yt_audio_out_fp, audio_file))
		
		## Compress
		# Compress the data using GZIP compression
		compressed_data = gzip.compress(summary_stats_feats.tobytes())

		# Create the PyArrow Array from the compressed data
		compressed_array = pa.array([compressed_data], pa.large_binary())

		# Create the PyArrow Table from the array
		table = pa.Table.from_arrays([compressed_array], names=['data'])

		# Create the output file path
		output_filepath = os.path.join(audio_data_fp, f'summary_stats_audio_feats_gzip_{i}.parquet')

		# Write the compressed data to a Parquet file
		with pq.ParquetWriter(output_filepath, table.schema) as writer:
			writer.write_table(table)
		
		# Insert fp to list
		audio_data_fp_list.append(output_filepath)
		
		
	# Grab num files
	parquet_out_file_list = os.listdir(audio_data_fp)
	num_parquet_out_files = len(parquet_out_file_list)

	print(f"{num_parquet_out_files} parquet files have been created in {audio_data_fp}")

	return audio_data_fp_list




def yt_metadata_extraction(yt_metadata_out_fp):
    ### Step 4: Metadata extraction and compile
    df_md = pd.DataFrame(columns=['track', 'artist', 'album'])
    
    json_file_list = os.listdir(yt_metadata_out_fp)
    for i, json_output_file in enumerate(json_file_list):
        ## Get metadata
        # Open the JSON file
        with open(os.path.join(yt_metadata_out_fp, json_output_file), 'r') as file:
            # Parse the JSON data
            json_data = json.load(file)

            keys = ['track', 'artist', 'album', 'webpage_url']
            result = {}

            for key in keys:
                try:
                    result[key] = json_data[key]
                except KeyError:
                    result[key] = None

            json_track, json_artist, json_album, json_yt_link = result['track'], result['artist'], result['album'], result['webpage_url']

        new_md_row = {
            'track': json_track,
            'artist': json_artist,
            'album': json_album,
            'yt_link': json_yt_link
        }

        new_md_row_df = pd.DataFrame([new_md_row])

        df_md = pd.concat([df_md, new_md_row_df], ignore_index=True)

    return df_md




def yt_df_assemble_save(df_md, df_file_path, playlist_url, playlist_description, yt_audio_data_base_fp, audio_data_fp_list):
	df_ref = df_md.copy()
	df_ref['stereo_gzip_mfcc_fp'] = None
	df_ref['playlist_url'] = playlist_url
	df_ref['playlist_description'] = playlist_description
	df_ref['playlist_fp'] = yt_audio_data_base_fp
	df_ref['dl_date'] = date.today().strftime("%Y-%m-%d")
	df_ref['mono_gzip_mfcc_fp'] = None
	df_ref['mono_gzip_summary_stats_audio_feats_fp'] = audio_data_fp_list


	## Save df
	# Create the file path
	save_df_file_path = fr"{df_file_path}\df.csv"

	# Save the DataFrame to the file path
	df_ref.to_csv(save_df_file_path, index=False)

	
	print(f"Dataframe saved in {save_df_file_path}")
	return df_ref




def yt_df_append_master(master_ref_df_fp, df_ref):
	df_master = pd.read_csv(master_ref_df_fp)
	
	## Add metadata to master df
	# Rename/shape columns in df_local to match df_master
	df_renamed = df_ref.rename(columns={
			'yt_link': 'song_origin',
			'playlist_description': 'run_origin',
			'dl_date': 'run_date'
		}).drop(columns = ['playlist_url'])
		
	df_renamed.reset_index(inplace=True)
	df_renamed.rename(columns={'index': 'original_index'}, inplace=True)
		
	df_renamed = df_renamed[df_master.columns.to_list()]
	
	## Append data
	df_master = df_master.append(df_renamed, ignore_index=True)
	
	return df_master




def yt_temp_file_cleanup(yt_audio_out_fp, yt_metadata_out_fp):
    # Audio:
    audio_file_list = os.listdir(yt_audio_out_fp)

    for audio_file in audio_file_list:
        audio_file_path = os.path.join(yt_audio_out_fp, audio_file)
        os.remove(audio_file_path)

    # Json:
    json_file_list = os.listdir(yt_metadata_out_fp)

    for json_file in json_file_list:
        json_file_path = os.path.join(yt_metadata_out_fp, json_file)
        os.remove(json_file_path)


    ## Config msg
    # Reset vars
    audio_file_list = os.listdir(yt_audio_out_fp)
    json_file_list = os.listdir(yt_metadata_out_fp)
    
    files_del_msg = f"{yt_audio_out_fp} and {yt_metadata_out_fp} are now empty" if len(audio_file_list) == 0 and len(json_file_list) == 0 else "Error: Files still exist"
    
    print(files_del_msg)