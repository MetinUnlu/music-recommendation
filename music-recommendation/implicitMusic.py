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

    def track_mapping(self,artists_df,interactin_df):
        track_id_to_index = {track_id: i for i, track_id in enumerate(interactin_df.track_id.unique())}
        artists_df['track']=artists_df['track_id'].map(track_id_to_index)
        return artists_df

    def get_artist_name_from_id(self, artist_id: int) -> str:
        return self._artists_df.loc[artist_id, "name"]
    def get_track_name_from_id(self, artist_id: int) -> str:
        return self._artists_df.loc[artist_id, "artist"]

    def load_artists(self, artists_file: Path, interaction_file: Path) -> None:
        """Load the artists file and stores it as a Pandas dataframe in a
        private attribute.
        """
        artists_df = pd.read_csv(artists_file)
        interaction_df=pd.read_csv(interaction_file)
        artists_df=self.track_mapping(artists_df,interaction_df)
        artists_df = artists_df.set_index("track")
        self._artists_df = artists_df

"""This module features the ImplicitRecommender class that performs
recommendation using the implicit library.
"""


from pathlib import Path
from typing import Tuple, List

import implicit
import scipy

# from musiccollaborativefiltering.data import load_user_artists, ArtistRetriever


class ImplicitRecommender:
    """The ImplicitRecommender class computes recommendations for a given user
    using the implicit library.

    Attributes:
        - artist_retriever: an ArtistRetriever instance
        - implicit_model: an implicit model
    """

    def __init__(
        self,
        artist_retriever: ArtistRetriever,
        implicit_model: implicit.recommender_base.RecommenderBase,
    ):
        self.artist_retriever = artist_retriever
        self.implicit_model = implicit_model

    def fit(self, user_artists_matrix: scipy.sparse.csr_matrix) -> None:
        """Fit the model to the user artists matrix."""
        self.implicit_model.fit(user_artists_matrix)

    def recommend(
        self,
        user_id: int,
        user_artists_matrix: scipy.sparse.csr_matrix,
        n: int = 10,
    ) -> Tuple[List[str], List[float]]:
        """Return the top n recommendations for the given user."""
        artist_ids, scores = self.implicit_model.recommend(
            user_id, user_artists_matrix[n], N=n
        )
        artists = [
            self.artist_retriever.get_artist_name_from_id(artist_id)
            for artist_id in artist_ids
        ]
        tracks = [
            self.artist_retriever.get_track_name_from_id(artist_id)
            for artist_id in artist_ids
        ]
        return artist_ids,artists, tracks, scores



