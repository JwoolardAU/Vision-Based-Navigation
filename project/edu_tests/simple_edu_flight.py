from djitellopy import Tello
from djitellopy import TelloSwarm
import time
# TELLO-CBF2F9 -- Tello-2 -- MAC- 60:60:1F:CB:F2:F9
# TELLO-CBF308 -- Tello-1 -- MAC- 60:60:1F:CB:F3:08


Tello1_IP = "192.168.1.100"
Tello2_IP = "192.168.1.200"

drone1 = Tello(Tello1_IP)
drone2 = Tello(Tello2_IP)

swarm = TelloSwarm([drone1,drone2])

swarm.connect()
swarm.takeoff()

swarm.move_up(100)
swarm.move_down(100)

swarm.flip_forward()

swarm.sequential(lambda i, tello: tello.flip_back())

swarm.parallel(lambda i, tello: tello.move_up(50))

swarm.sequential(lambda i, tello: tello.curve_xyz_speed(30,0,10, 50,0,10,20))

swarm.parallel(lambda i, tello: tello.rotate_clockwise(180))

swarm.parallel(lambda i, tello: tello.curve_xyz_speed(30,0,10, 50,0,10,20))

swarm.land()
swarm.end()


### RC Control Tests ###
# drone1 = Tello("192.168.1.50")
# drone1.connect()

# drone1.takeoff()

# drone1.send_rc_control(-15,-15,0,0)

# time.sleep(10)
# drone1.send_rc_control(0,0,0,0)
### End RC Control Tests ###

# # swarm.sequential(lambda i, tello: tello.move_forward(i * 20 + 20))


