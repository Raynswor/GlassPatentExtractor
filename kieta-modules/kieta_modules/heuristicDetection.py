from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from kieta_data_objs import Area, BoundingBox, Document, GroupedAreas

from . import Module


@dataclass
class CandidateCluster:
    def __init__(self, start: int, end: int, lines: List[GroupedAreas] = None) -> None:
        self.start: int = start
        self.end: int = end
        self.lines: List[GroupedAreas] = lines if lines else list()
    
    def __len__(self) -> int:
        return self.end - self.start + 1
    
    def max_size(self) -> int:
        return max([len(l) for l in self.lines])
    
    def min_size(self) -> int:
        return min([len(l) for l in self.lines])

    def append(self, obj) -> None:
        self.end += 1
        self.lines += [obj]
    
    def __repr__(self) -> str:
        return f"{self.start}-{self.end} (#lines:{len(self)},min({self.min_size()}),max({self.max_size()}))"
    
    def get_boundingBox(self) -> BoundingBox: 
        bb = BoundingBox(10000, 10000, 0, 0)
        for vv in self.lines:
            bb.x1 = min(vv.get_boundingBox().x1, bb.x1)
            bb.y1 = min(vv.get_boundingBox().y1, bb.y1)
            bb.x2 = max(vv.get_boundingBox().x2, bb.x2)
            bb.y2 = max(vv.get_boundingBox().y2, bb.y2)
        return bb
        

def get_alignments(lines: List[GroupedAreas], tolerance) -> List[Tuple[float, float, float]]:
    rets: List[List[float, float, float]] = []
    for l in lines:
        for a in l.areas:
            added = False
            for r in rets:
                # check if there is alignment
                if a.boundingBox.x1-tolerance <= r[0] <= a.boundingBox.x1+tolerance or \
                    a.boundingBox.x2-tolerance <= r[2] <= a.boundingBox.x2+tolerance or \
                    0.5*(a.boundingBox.x1+a.boundingBox.x2)-tolerance <= r[1] <= 0.5*(a.boundingBox.x1+a.boundingBox.x2)+tolerance:
                    added = True
                    break
            if not added:
                rets.append([a.boundingBox.x1, 0.5*(a.boundingBox.x1+a.boundingBox.x2), a.boundingBox.x2])
            
    return rets

def merge_cluster(l: List[CandidateCluster]):
    cc = CandidateCluster(10000,0,[])
    for ll in l:
        cc.start = min(cc.start, ll.start)
        cc.end = max(cc.end, ll.end)
        cc.lines += ll.lines
    return cc


def group_horizontally_by_distance(areas: List[Area], threshold_horizontal: int, threshold_height_diff: int, threshold_vertical_translation: int) -> List[GroupedAreas]:
    # Initialize empty dict to store groups
    groups: List[GroupedAreas] = []
    # Iterate over each object
    for obj in areas:
        done = False
        # ignore objects that are higher than wide 
        if (t:=obj.boundingBox.y2-obj.boundingBox.y1) > obj.boundingBox.x2-obj.boundingBox.x1 and t > 30:
            continue

        # Iterate over other objects
        for other_group in groups:
            for other in other_group.areas:
                if obj.oid == other.oid or other.boundingBox.y2-other.boundingBox.y1 > other.boundingBox.x2-other.boundingBox.x1:
                    continue
                # check if same vertical position
                if abs(obj.boundingBox.y1-threshold_vertical_translation <= other.boundingBox.y1 <= obj.boundingBox.y1+threshold_vertical_translation) and \
                abs(obj.boundingBox.y2-threshold_vertical_translation <= other.boundingBox.y2 <= obj.boundingBox.y2+threshold_vertical_translation):
                    # Calculate horizontal distance between objects
                    if abs(obj.boundingBox.x1 - other.boundingBox.x2) < threshold_horizontal or abs(obj.boundingBox.x2 - other.boundingBox.x1) < threshold_horizontal:
                        # calculate vertical translation
                        if abs((obj.boundingBox.y2-obj.boundingBox.y1)-(other.boundingBox.y2-other.boundingBox.y1)) < threshold_height_diff:
                            other_group.areas.append(obj)
                            done = True
                            break
            if done:
                break
        if not done:
            groups.append(GroupedAreas([obj]))
    groups.sort(key=lambda x: x.get_boundingBox().y1)

    return groups


def scoring_algorithm(buffer: List[CandidateCluster], context: List[GroupedAreas], irregular_row_skips: int, distance_between_rows: float, alignment_tolerance: float):
    def check_alignment(c1: CandidateCluster, c2: CandidateCluster) -> bool:
        c1_aligs = get_alignments(c1.lines, alignment_tolerance)
        c2_aligs = get_alignments(c2.lines, alignment_tolerance)

        found = 0
        for r in c1_aligs:
            for r2 in c2_aligs:
                # check if there is alignment
                if r2[0]-alignment_tolerance <= r[0] <= r2[0]+alignment_tolerance or \
                    r2[1]-alignment_tolerance <= r[1] <= r2[1]+alignment_tolerance or \
                    r2[2]-alignment_tolerance <= r[2] <= r2[2]+alignment_tolerance:
                    found += 1
                    break
        return found == len(c1_aligs) or found == len(c2_aligs)
    # compare with lines above and below
    # scoring
    # 100 P for 
    
    # return buffer[0].max_size() == buffer[-1].max_size() and \
    return  check_alignment(buffer[0], buffer[-1]) and \
        abs(buffer[0].end-buffer[-1].start) <= irregular_row_skips+1 and \
            abs(buffer[0].get_boundingBox().y2-buffer[-1].get_boundingBox().y1) < distance_between_rows


class HeuristicDetector(Module):
    _MODULE_TYPE = 'HeuristicDetector'
    def __init__(self, stage: int, parameters: Optional[Dict] = {}, debug_mode: bool = False) -> None:
        super().__init__(stage, parameters, debug_mode)
        self.whitespace_pixel_size: int = int(parameters["whitespace_pixel_size"])
        self.threshold_height_diff: int = int(parameters["threshold_height_diff"])
        self.threshold_vertical_translation: int = int(parameters["threshold_vertical_translation"])
        # self.maximal_line_gap: int = int(parameters("maximal_line_gap"))
        self.irregular_row_skips: int = int(parameters["irregular_row_skips"])
        self.distance_between_rows: int = int(parameters["distance_between_rows"])
        self.alignment_tolerance: int = int(parameters["alignment_tolerance"])


    def execute(self, inpt: Document) -> Document:
        # zeilenweise gruppieren
        # gruppen pro zeile erkennen
        # --> nimm XMLTextLines, die sind schon relativ gut gruppiert
        # scoring verfahren -> #gruppen pro Zeile (aka Spalten)
        # wenn in irgendeiner Weise passend, als Kandidat verwenden
        # schauen ob merge mit oben unten moeglich ist (selbe Anzahl an Spalten)
        # eventuell gap zulassen, wenn mal Zeilen mit weniger drinnen sind
        # verbinden mit eingabe: ist mehrspaltiges Dokument, Freiform, einspaltig

        for k, v in inpt.pages.byId.items():  # per page
            areas_on_this_page: List[Area] = [inpt.areas.byId[area] for area in inpt.references.byId[k]]  # per area
            
            grouped: List[GroupedAreas] = group_horizontally_by_distance(filter(lambda x: x.category=="XMLElementLine", areas_on_this_page), 1000, self.threshold_height_diff, self.threshold_vertical_translation)

            candidates = [CandidateCluster(start=ix, end=ix,lines=[grouped[ix]]) for ix in range(len(grouped)) if len(grouped[ix]) > 1]
            # merge if number of irregular lines between groups is smaller than threshold and both parts have same number of cells
            # remove others
            merged_candidates = list()
            buffer: List[CandidateCluster] = list()
            candidates_bck = [x for x in candidates]

            # erstes gegen jedes von self.irregular lineskips+2
            # falls ja, merge
            # falls nein pop und rein
            try:
                while (t:= candidates.pop(0)):
                    buffer.append(t)
                    if len(buffer) < 2:
                        continue
                    # here test algorithm
                    if scoring_algorithm(buffer, candidates_bck, self.irregular_row_skips, self.distance_between_rows, self.alignment_tolerance):
                    # if buffer[0][-1][1] == buffer[-1][0][1] and abs(buffer[0][-1][0]-buffer[-1][0][0]) <= self.irregular_line_skips+1:
                        buffer = [merge_cluster(buffer)]

                    if len(buffer) == self.irregular_row_skips+2:
                        merged_candidates.append(buffer.pop(0))
                        candidates = buffer + candidates
                        buffer.clear()
                    
                    if not candidates:
                        merged_candidates.extend(buffer)
            except IndexError:
                if buffer[0].max_size() == buffer[-1].max_size():
                    merged_candidates.extend([merge_cluster(buffer)])
                else:
                    merged_candidates.extend(buffer)
            
            # merge if there are two groups that overlap in any way
            new_merged = []
            while (current := merged_candidates.pop(0)):
                added = False
                for other in range(len(new_merged)):
                    if current.get_boundingBox().overlap(new_merged[other].get_boundingBox()):
                        added = True
                        new_merged.append(merge_cluster([current, new_merged.pop(other)]))
                        break
                if not added:
                    new_merged.append(current)
                if len(merged_candidates) == 0:
                    break
            
            for group in new_merged:
                if group.max_size() == 1:
                    continue
                inpt.add_area(k, "Table", group.get_boundingBox())

        return inpt


