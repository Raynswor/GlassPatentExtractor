import re
from typing import Any, Dict, List, Optional, Tuple, NamedTuple
from kieta_modules.base import Module
from kieta_modules import util

from kieta_data_objs import Document, BoundingBox, Area

from concurrent.futures import ThreadPoolExecutor

import Levenshtein


class DictionaryCorrectorModule(Module):
    _MODULE_TYPE = 'DictionaryCorrectorModule'

    def __init__(self, stage: int, parameters: Optional[Dict] = None, debug_mode: bool = False) -> None:
        super().__init__(stage, parameters, debug_mode)

        self.dict_path = parameters.get('dict_path', "")
        self.trie: Trie = Trie()
        self.load_dict()

        self.distance = parameters.get("distance", 1)
        self.distance_type = parameters.get("distance_type", "levenshtein")

        self.number_regex = re.compile(r'^(?<!\d)[+-]?(\d{1,3}(?:[.,]\d{3})*|[.,]?\d+)(?=[.,]\d{1,2})?[.,]?\d*(?!\d)')
    
    def load_dict(self):
        with open(self.dict_path, 'r') as f:
            for line in f:
                line = line.strip()
                self.trie.insert(line)

    def execute(self, inpt: Document) -> Document:
        areas = [inpt.get_area_type(x) for x in self.apply_to]
        # flatten
        areas = [x for y in areas for x in y]

        with ThreadPoolExecutor() as executor:
            futures = []
            for area in areas:
                if area.data and 'content' in area.data:
                    future = executor.submit(self.correct, area.data['content'])
                    futures.append(future)

            for future, area in self.get_progress_bar(zip(futures, areas), desc="Correcting", unit="areas", total=len(areas)):
                if future.result() is not None:
                    area.data['content'] = future.result()

        return inpt
    
    def correct(self, text: str) -> str:
        # is number
        if re.search(self.number_regex, text):
            return text
        if self.trie.query(text):
            return text
        candidates = self.get_candidates(text)
        if len(candidates) == 0:
            return text
        return candidates[0]
    
    def get_candidates(self, text: str) -> List[str]:
        candidates = set()
        counter = len(text)-1
        while counter >= 1:
            for x in self.trie.query(text[:counter]):
                candidates.add(x)
        
        print(f"candidates: {len(candidates)} for {text}")

        for key in candidates:
            if self.distance_type == "levenshtein":
                if  Levenshtein.distance(text, key) <= self.distance:
                    candidates.append(key)
            # elif self.distance_type == "damerau":
            #     if util.damerau_levenshtein(text, key) <= self.distance:
            #         candidates.append(key)
            else:
                raise ValueError("Unknown distance type")
        return candidates
    



class TrieNode:
    """A node in the trie structure"""

    def __init__(self, char):
        # the character stored in this node
        self.char = char

        # whether this can be the end of a word
        self.is_end = False

        # a counter indicating how many times a word is inserted
        # (if this node's is_end is True)
        self.counter = 0

        # a dictionary of child nodes
        # keys are characters, values are nodes
        self.children = {}

class Trie(object):
    """The trie object"""

    def __init__(self):
        """
        The trie has at least the root node.
        The root node does not store any character
        """
        self.root = TrieNode("")
    
    def insert(self, word):
        """Insert a word into the trie"""
        node = self.root
        
        # Loop through each character in the word
        # Check if there is no child containing the character, create a new child for the current node
        for char in word:
            if char in node.children:
                node = node.children[char]
            else:
                # If a character is not found,
                # create a new node in the trie
                new_node = TrieNode(char)
                node.children[char] = new_node
                node = new_node
        
        # Mark the end of a word
        node.is_end = True

        # Increment the counter to indicate that we see this word once more
        node.counter += 1
        
    def dfs(self, node, prefix):
        """Depth-first traversal of the trie
        
        Args:
            - node: the node to start with
            - prefix: the current prefix, for tracing a
                word while traversing the trie
        """
        if node.is_end:
            self.output.append((prefix + node.char, node.counter))
        
        for child in node.children.values():
            self.dfs(child, prefix + node.char)
        
    def query(self, x):
        """Given an input (a prefix), retrieve all words stored in
        the trie with that prefix, sort the words by the number of 
        times they have been inserted
        """
        # Use a variable within the class to keep all possible outputs
        # As there can be more than one word with such prefix
        self.output = []
        node = self.root
        
        # Check if the prefix is in the trie
        for char in x:
            if char in node.children:
                node = node.children[char]
            else:
                # cannot found the prefix, return empty list
                return []
        
        # Traverse the trie to get all candidates
        self.dfs(node, x[:-1])

        # Sort the results in reverse order and return
        return sorted(self.output, key=lambda x: x[1], reverse=True)