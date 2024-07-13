"""This module features functions and classes to manipulate data for the
collaborative filtering algorithm.
"""

from pathlib import Path

import scipy
import pandas as pd


def str_to_index(df,user_id='user_id',track_id='track_id'):
    # Create a mapping from user IDs and track IDs to numeric indices
    user_id_to_index = {user_id: i for i, user_id in enumerate(df.user_id.unique())}
    track_id_to_index = {track_id: i for i, track_id in enumerate(df.track_id.unique())}

    # Replace the strings with numeric indices
    df['user'] = df.user_id.map(user_id_to_index)
    df['track'] = df.track_id.map(track_id_to_index)

    df.set_index(["user", "track"], inplace=True)

    return df

def load_user_artists(user_artists_file: Path) -> scipy.sparse.csr_matrix:
    """Load the user artists file and return a user-artists matrix in csr
    fromat.
    """
    user_artists = str_to_index(pd.read_csv(user_artists_file).drop(['Unnamed: 0'], axis=1))
    # user_artists.set_index(["user_id", "track_id"], inplace=True)
    coo = scipy.sparse.coo_matrix(
        (
            user_artists.normalized_playcount.astype(float),
            (
                user_artists.index.get_level_values(0),
                user_artists.index.get_level_values(1),
            ),
        )
    )
    return coo.tocsr()


class ArtistRetriever:
    """The ArtistRetriever class gets the artist name from the artist ID."""

    def __init__(self):
        self._artists_df = None

    def get_artist_name_from_id(self, artist_id: int) -> str:
        """Return the artist name from the artist ID."""
        return self._artists_df.loc[artist_id, "name"]

    def load_artists(self, artists_file: Path) -> None:
        """Load the artists file and stores it as a Pandas dataframe in a
        private attribute.
        """
        artists_df = pd.read_csv(artists_file, sep="\t")
        artists_df = artists_df.set_index("id")
        self._artists_df = artists_df