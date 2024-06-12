# Kieta Data Objs
This module contains the internal data structure used in the KIETA pipeline

# Important Classes
The following paragraphs describe the most important classes used within the pipeline.

## DocObj (base.py)
The DocObj class is an abstract base class (ABC) that defines a common interface for all document objects. It declares two abstract methods: from_dic and to_dic.

### Methods
from_dic: A static method that takes a dictionary as input and returns an instance of the class. The exact behavior is defined in each subclass.
to_dic: An instance method that returns a dictionary representation of the instance. The exact behavior is defined in each subclass.

## Document (document.py)
Class represents a document with several properties:

### Attributes
- oid: A unique identifier for the document.
- pages: A NormalizedObj of Page objects representing the pages in the document.
- areas: A NormalizedObj of Area objects representing areas in the document.
- links: A NormalizedObj of Link objects representing links in the document.
- references: A NormalizedObj of lists of strings representing references in the document.
- revisions: A list of Revision objects representing revisions of the document.
- fonts: A list of Font objects representing fonts used in the document.
- onto_information: A list of Entity objects representing ontology information of the document.
- metadata: A dictionary containing any additional metadata of the document.
- raw_pdf: A bytes object representing the raw PDF data of the document.

## BoundingBox (geometry.py)
The BoundingBox class represents a rectangular area defined by two points in a 2D space. The points are represented by their x and y coordinates. The class ensures that the bounding box is valid by setting any negative coordinates to 0 and making sure that the first point is always the top-left corner and the second point is always the bottom-right corner.

### Attributes
x1, y1: Coordinates of the top-left corner of the bounding box.
x2, y2: Coordinates of the bottom-right corner of the bounding box.
_img_sp: A boolean flag indicating whether the bounding box is in image space.


## NormalizedObj (normalizedObject.py)
The NormalizedObj class is a generic class that represents a normalized object. It contains a dictionary byId that maps string keys to objects of the generic type T. It also maintains a list of all keys in the dictionary.

### Attributes
byId: A dictionary mapping string keys to objects of type T.
allIds: A list of all keys in the byId dictionary.
