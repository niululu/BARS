// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract PhysioChain {
    struct Record {
        bytes32 commitment;
        string did;
        bool revoked;
        uint256 timestamp;
    }

    Record[] public records;

    event Registered(uint256 indexed id, bytes32 commitment, string did);
    event Revoked(uint256 indexed id);

    function register(bytes32 commitment, string calldata did)
        external
        returns (uint256)
    {
        records.push(Record(commitment, did, false, block.timestamp));
        uint256 id = records.length - 1;
        emit Registered(id, commitment, did);
        return id;
    }

    function revoke(uint256 id) external {
        require(id < records.length, "Invalid ID");
        records[id].revoked = true;
        emit Revoked(id);
    }

    function getRecord(uint256 id)
        external
        view
        returns (bytes32, string memory, bool, uint256)
    {
        require(id < records.length, "Invalid ID");
        Record memory r = records[id];
        return (r.commitment, r.did, r.revoked, r.timestamp);
    }
}
