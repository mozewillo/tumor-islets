from scipy.spatial import distance
from enum import Enum


class MarkerIF1(Enum):
    # IF 1
    CD11c = 0
    CD15 = 1
    CD163 = 2
    CD20 = 3
    CD3 = 4
    CK = 5


class MarkerIF2(Enum):
    # IF 2
    CD8 = 0
    CK = 1
    GB = 2
    Ki67 = 3
    PD1 = 4
    PDL1 = 5


class MarkerIF3(Enum):
    # IF 3
    CD3 = 0
    CD4 = 1
    CD56 = 2
    CD8 = 3
    CK = 4
    FOXP3 = 5


Marker = {'1': MarkerIF1, '2': MarkerIF2, '3': MarkerIF3}


class Cell(object):
    """Provides an object to wrap cell data
    Attributes:
        x                x-axis location of a cell (um)
        y                y-axis location of a cell (um)
        scores           dictionary {marker: float} of marker continuous activity values
        tissue_type      tissue type to which the cell belongs [Cancer, Stroma]
        phenotype_label  string of all active marker names in cell separated by dash (-)
        activities_tuple tuple of activities of markers: CK, CD3, CD11c, CD15, CD20, CD163
        is_ck            True if activity of CK > 1, else False
    """
    __slots__ = ['id', 'x', 'y', 'phenotype_label', 'is_ck', 'tissue_type', "phenotype",
                 'activities_tuple', 'scores', 'phenotype_original', "in_ROI_tumor_tissue"]

    def __init__(self, record, idx, panel):
        """
        Parameters
        ----------
        record : DataFrame
            DataFrame containing one record of description of IF panel cell

        Returns
        -------
        Cell
            Cell object with that includes:
                x -
                y -
        """
        self.id = idx
        self.x = float(record['nucleus.x'])
        self.y = float(record['nucleus.y'])

        # raw markers scores
        self.scores = dict([(i.split(".")[0], float(record[i]))
                            for i in sorted(record.keys()) if "score.normalized" in i])

        # binary marker activities 
        _activities = dict([(i.split(".")[0], float(record[i]) > 1)
                            for i in sorted(record.keys()) if "score.normalized" in i])
        self.activities_tuple = tuple([_activities[m.name] for m in Marker[panel]])
        self.is_ck = _activities["CK"]
        if "in.ROI.tumor_tissue" in record.keys():
            self.in_ROI_tumor_tissue = record["in.ROI.tumor_tissue"]
        self.tissue_type = record['tissue.type']

        # derived cell phenotype: active markers separated with dash, e.g. "CK-CD15" 
        _phenotypes = [key for key, value in _activities.items() if value]

        # sort phenotype alphabetically
        # self.phenotype = ""
        # for key in sorted(_activities.keys()):
        #     self.phenotype += key
        #     if _activities[key]:
        #         self.phenotype += "+"
        #     else:
        #         self.phenotype += "-"

        # delete phenotype label and change in code!
        self.phenotype_label = '-'.join(sorted(_phenotypes, reverse=True))
        if self.phenotype_label == "":
            self.phenotype_label = "neg"

        # cell phenotype from the raw data file, e.g. "CK+CD3-CD11c-CD15-CD163+CD20-"
        self.phenotype_original = record["phenotype"]

    def __eq__(self, other):
        """Compares two cells - two cells are equal if they have the same location (x and y)
        and the same phenotype_label."""
        if isinstance(other, Cell):
            return self.x == other.x and self.y == other.y \
                   and self.phenotype_label == other.phenotype_label
        return False

    def __hash__(self):
        return hash(str(self))

    @property
    def position(self):
        """The radius property."""
        return self.x, self.y

    def marker_is_active(self, marker):
        """Returns boolean value of marker activity."""
        return self.activities_tuple[marker.value]

    def marker_is_active_str(self, marker_name: str):
        """check if given marker name is in marker set for the cell and if the marker is active"""
        for m in Marker:
            if m.name == marker_name:
                return self.activities_tuple[m.value]
        print(f"Marker {marker_name} not found")
        return 1

    def distance(self, other_cell):
        """Calculates euclidean distance between from the other_cell
        Parameters
        ----------
        other_cell : Cell
            a cell to which a distance needs to be calculated
        Returns
        -------
        distance
            distance from the cell to other_cell
        """
        return distance.euclidean((self.x, self.y), (other_cell.x, other_cell.y))

    def __str__(self):
        """Function to pretty printing using __str__(), print(object), print(str(object))"""
        return "" + self.phenotype_label + " at:" + " (" + str(self.x) + ", " + str(self.y) + ")"
