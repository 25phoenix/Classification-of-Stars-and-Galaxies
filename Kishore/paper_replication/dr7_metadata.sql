SELECT DISTINCT p.objID AS objid, s.z AS z, p.petroMag_r AS petromag_r
INTO MyDB.barchi_dr7
FROM SpecObj AS s JOIN PhotoObjAll AS p ON s.bestObjID = p.objID
WHERE s.bestObjID > 0
