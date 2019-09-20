import copy
import logging
from abc import abstractmethod
from collections import defaultdict
from typing import (DefaultDict, Dict, Generic, List, Optional, Set, Type,
                    TypeVar, Union, Tuple, Hashable)

import jsonpickle

from forte.data.ontology import (Annotation, Entry, EntryType, Group,
                                 Link, Span, BaseLink)

logger = logging.getLogger(__name__)

__all__ = [
    "BasePack",
    "BaseMeta",
    "InternalMeta",
    "BaseIndex",
    "PackType"
]


class BaseMeta:
    """
    Basic Meta information for both DataPack and MultiPack.
    """

    def __init__(self, doc_id: Optional[str] = None):
        self.doc_id: str = doc_id

        # TODO: These two are definitely internal.
        self.process_state: str = ''
        self.cache_state: str = ''


class InternalMeta:
    """
    The internal meta information of **one kind of entry** in a datapack.
    Note that the :attr:`internal_metas` in :class:`BasePack` is a dict in
    which the keys are entries types and the values are objects of
    :class:`InternalMeta`.
    """

    def __init__(self):
        self.id_counter = 0
        self.fields_created = defaultdict(set)
        self.default_component = None

        # TODO: Finish the update of this true component_records.
        # A index of the component records of entries and fields. These will
        # indicate "who" created the entry and modified the fields.
        self.component_records: Dict[
            str,  # The component name.
            Set[int],  # The set of entries created by this component.
            Set[  # The set of fields created by this component.
                Tuple[int, str]  # The 2-tuple identify the entry field.
            ]
        ]



class BasePack:
    """
    The base class of DataPack and MultiPack
    """

    def __init__(self, doc_id: Optional[str] = None):
        self.links: List[Link] = []
        self.groups: List[Group] = []

        self.meta: BaseMeta = BaseMeta(doc_id)
        self.index: BaseIndex = BaseIndex(self)
        self.internal_metas: \
            Dict[type, InternalMeta] = defaultdict(InternalMeta)

        # This is used internally when a processor takes the ownership of this
        # DataPack.
        self.__owner_component = None

    def enter_processing(self, component_name: str):
        self.__owner_component = component_name

    def current_component(self):
        return self.__owner_component

    def exit_processing(self):
        self.__owner_component = None

    def set_meta(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self.meta, k):
                raise AttributeError(f"Meta has no attribute named {k}")
            setattr(self.meta, k, v)

    @abstractmethod
    def add_entry(self, entry: EntryType) -> EntryType:
        """
        Add an :class:`Entry` object to the :class:`BasePack` object.
        Allow duplicate entries in a pack.

        Args:
            entry (Entry): An :class:`Entry` object to be added to the pack.

        Returns:
            The input entry itself
        """
        raise NotImplementedError

    @abstractmethod
    def add_or_get_entry(self, entry: EntryType) -> EntryType:
        """
        Try to add an :class:`Entry` object to the :class:`DataPack` object.
        If a same entry already exists, will return the existing entry
        instead of adding the new one. Note that we regard two entries to be
        same if their :meth:`eq` have the same return value, and users could
        override :meth:`eq` in their custom entry classes.

        Args:
            entry (Entry): An :class:`Entry` object to be added to the pack.

        Returns:
            If a same entry already exists, returns the existing
            entry. Otherwise, return the (input) entry just added.
        """
        raise NotImplementedError

    @abstractmethod
    def record_fields(self, fields: List[str], entry_type: Type[Entry],
                      component: str):
        """Record in the internal meta that the ``entry_type`` generated by
        ``component`` have ``fields``.

        If ``component`` is "_ALL_", we will record ``fields`` for all existing
        components in the internal meta of ``entry_type``.
        """
        raise NotImplementedError

    def serialize(self) -> str:
        """
        Serializes a pack to a string.
        """
        return jsonpickle.encode(self, unpicklable=True)

    def view(self):
        return copy.deepcopy(self)


PackType = TypeVar('PackType', bound=BasePack)


class BaseIndex(Generic[PackType]):
    """
    A set of indexes used in a datapack:

    (1) :attr:`entry_index`,
    the index from each tid to the corresponding entry;

    (2) :attr:`type_index`, the index from each type to the entries of that
    type;

    (3) :attr:`component_index`, the index from each component to the
    entries generated by that component;

    (4) :attr:`link_index`, the index
    from child (:attr:`link_index["child_index"]`)and parent
    (:attr:`link_index["parent_index"]`) nodes to links;

    (5) :attr:`group_index`, the index from group members to groups.
    """

    def __init__(self, data_pack):
        self.data_pack: PackType = data_pack
        # basic indexes (switches always on)
        self.entry_index: Dict[str, Entry] = dict()
        self.type_index: DefaultDict[Type, Set[str]] = defaultdict(set)
        self.component_index: DefaultDict[str, Set[str]] = defaultdict(set)
        # other indexes (built when first looked up)
        self._group_index = defaultdict(set)
        self._link_index: Dict[str, DefaultDict[Hashable, set]] = dict()
        # indexing switches
        self._group_index_switch = False
        self._link_index_switch = False

    @property
    def link_index_switch(self):
        return self._link_index_switch

    def turn_link_index_switch(self, on: bool):
        self._link_index_switch = on

    @property
    def group_index_switch(self):
        return self._group_index_switch

    def turn_group_index_switch(self, on: bool):
        self._group_index_switch = on

    def link_index(self, tid: str, as_parent: bool = True) -> Set[str]:
        """
        Look up the link_index with key ``tid``.

        Args:
            tid (str): the tid of the entry being looked up.
            as_parent (bool): If `as_patent` is True, will look up
                :attr:`link_index["parent_index"] and return the tids of links
                whose parent is `tid`. Otherwise,  will look up
                :attr:`link_index["child_index"] and return the tids of links
                whose child is `tid`.
        """
        if not self._link_index_switch:
            self.update_link_index(self.data_pack.links)
        if as_parent:
            return self._link_index["parent_index"][tid]
        else:
            return self._link_index["child_index"][tid]

    def group_index(self, tid: str) -> Set[str]:
        """
        Look up the group_index with key `tid`.
        """
        if not self._group_index_switch:
            self.update_group_index(self.data_pack.groups)
        return self._group_index[tid]

    def update_basic_index(self, entries: List[Entry]):
        """Build or update the basic indexes, including

        (1) :attr:`entry_index`,
        the index from each tid to the corresponding entry;

        (2) :attr:`type_index`, the index from each type to the entries of that
        type;

        (3) :attr:`component_index`, the index from each component to the
        entries generated by that component.

        Args:
            entries (list): a list of entires to be added into the basic index.
        """
        for entry in entries:
            self.entry_index[entry.tid] = entry
            self.type_index[type(entry)].add(entry.tid)
            self.component_index[entry.component].add(entry.tid)

    def update_link_index(self, links: List[BaseLink]):
        """Build or update :attr:`link_index`, the index from child and parent
        nodes to links. :attr:`link_index` consists of two sub-indexes:
        "child_index" is the index from child nodes to their corresponding
        links, and "parent_index" is the index from parent nodes to their
        corresponding links.

        Args:
            links (list): a list of links to be added into the index.
        """
        logger.debug("Updating link index")
        if not self.link_index_switch:
            self.turn_link_index_switch(on=True)
            self._link_index["child_index"] = defaultdict(set)
            self._link_index["parent_index"] = defaultdict(set)
            links = self.data_pack.links

        for link in links:
            self._link_index["child_index"][
                link.child.index_key
            ].add(link.tid)
            self._link_index["parent_index"][
                link.parent.index_key
            ].add(link.tid)

    def update_group_index(self, groups: List[Group]):
        """Build or update :attr:`group_index`, the index from group members
         to groups.

        Args:
            groups (list): a list of groups to be added into the index.
        """
        logger.debug("Updating group index")
        if not self.group_index_switch:
            self.turn_group_index_switch(on=True)
            self._group_index = defaultdict(set)
            groups = self.data_pack.groups

        for group in groups:
            for member in group.members:
                self._group_index[member].add(group.tid)
