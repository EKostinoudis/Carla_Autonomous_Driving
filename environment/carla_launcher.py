import subprocess
import time
import logging

logger = logging.getLogger(__name__)

class CarlaLauncher():
    def __init__(self, port, launch_script, restart_after=-1, sleep=5.):
        '''
        port: The port of carla server
        launch_script: A script that takes as argument the port and launches the
            carla server
        restart_after: restart the server after every given value, if it is 
            negative never restart
        '''
        self.launch_script = launch_script + f' {port}'
        self.restart_after = restart_after
        self.sleep = sleep
        self.server = None
        self.count_resets = 0

        self.lauch()

        # TODO: maybe try to connect to the server and set it on sync mode, in
        #       order to avoid this sleep
        # extra sleep time for the cold start, also we may lauch many servers
        time.sleep(self.sleep * 20)

    def reset(self):
        if self.restart_after >= 0:
            if self.count_resets >= self.restart_after:
                self.lauch()
                self.count_resets = 0
                return

        self.count_resets += 1

    def lauch(self):
        self.kill()

        logger.info(f'Lauching server: {self.launch_script}')
        self.server = subprocess.Popen(
            self.launch_script,
            shell=True,
        )
        time.sleep(self.sleep)


    def kill(self):
        if self.server is not None:
            logger.info('Killing server')
            self.server.kill()
            time.sleep(self.sleep)

