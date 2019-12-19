# Docker image for OpenVSwitch
A Docker image for OpenVSwitch (version 2.12.0) http://www.openvswitch.org/

This OVS image runs only on Ubuntu 16.04 or 18.04 (as the underlying system, i.e., the OS on which OVS containers run).

### Instructions

##### Build the image
`docker build -t activeeon/ovs-ovn dockerfile`

##### Run only ovsdb-server
```
docker run -d --name=ovs-1 --cap-add=NET_ADMIN --cap-add=SYS_MODULE --cap-add=SYS_NICE --network=host --volume=/lib/modules:/lib/modules --security-opt label=disable --privileged activeeon/ovs-ovn ovsdb-server
```

##### Run only ovs-vswitchd
```
docker run -d --name=ovs-1 --cap-add=NET_ADMIN --cap-add=SYS_MODULE --cap-add=SYS_NICE --network=host --volume=/lib/modules:/lib/modules --security-opt label=disable --privileged activeeon/ovs-ovn ovs-vswitchd
```

##### Run both (ovsdb-server and ovs-vswitchd)
```
docker run -d --name=ovs-1 --cap-add=NET_ADMIN --cap-add=SYS_MODULE --cap-add=SYS_NICE --network=host --volume=/lib/modules:/lib/modules --security-opt label=disable --privileged activeeon/ovs-ovn ovs-all
```

##### Check status of ovsdb-server and ovs-vswitchd
`docker exec -it ovs-1 /usr/share/openvswitch/scripts/ovs-ctl status`

<br />
<br />
<br />
<br />

credits: 
- https://github.com/servicefractal/ovs
- https://hub.docker.com/r/shivarammysore/ovs
