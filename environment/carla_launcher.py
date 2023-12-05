import subprocess
import time

class CarlaLauncher():
    def __init__(self, port, launch_script, restart_after=-1, sleep=5.):
        '''
        port: The port of carla server
        launch_script: A script that takes as argument the port and launches the
            carla server
        restart_after: restart the server after every given value, if it is 
            negative never restart
        '''
        self.port = port
        self.launch_script = launch_script
        self.restart_after = restart_after
        self.sleep = sleep
        self.server = None
        self.count_resets = 0

    def reset(self):
        if self.restart_after >= 0:
            if self.count_resets >= self.restart_after:
                self.lauch()
                self.count_resets = 0
                return

        self.count_resets += 1

    def lauch(self):
        self.kill()

        self.server = subprocess.Popen(
            [self.launch_script, self.port],
            shell=True,
        )
        time.sleep(self.sleep)


    def kill(self):
        if self.server is not None:
            self.server.kill()
        time.sleep(self.sleep)

