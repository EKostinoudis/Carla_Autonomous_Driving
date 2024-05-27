import subprocess
import time
import logging
import carla
import psutil

logger = logging.getLogger(__name__)

class CarlaLauncher():
    def __init__(self, port, launch_script, restart_after=-1, sleep=10., launch_on_init=True, device=None):
        '''
        port: The port of carla server
        launch_script: A script that takes as argument the port and launches the
            carla server
        restart_after: restart the server after every given value, if it is 
            negative never restart
        sleep: time in seconds to sleep
        launch_on_init: if true, lanches the server in the init
        device: the device (gpu) to pass to the launch script
        '''
        self.device = device
        self.launch_script_base = launch_script
        self.update_port(port)

        self.restart_after = restart_after
        self.sleep = sleep
        self.server = None
        self.count_resets = 0
        self.port = port

        if launch_on_init:
            self.lauch()

    def reset(self, restart_server=False):
        if self.restart_after >= 0 or restart_server:
            if self.count_resets >= self.restart_after or restart_server:
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
        self.try_set_synchronous_mode()


    def kill(self):
        if self.server is not None:
            logger.info(f'Killing server: {self.port}')

            parent = psutil.Process(self.server.pid)
            for child in parent.children(recursive=True):
                child.kill()
            parent.kill()

            time.sleep(self.sleep)

    def update_port(self, port):
        self.launch_script = self.launch_script_base + f' {port}'
        if self.device is not None:
            self.launch_script += f' {self.device}'
        self.port = port


    def set_synchronous_mode(self, client, synchronous_mode=True, delta_seconds=0.1):
        settings = client.get_world().get_settings()
        settings.synchronous_mode = synchronous_mode
        settings.fixed_delta_seconds = delta_seconds
        client.get_world().apply_settings(settings)

    def try_set_synchronous_mode(self, tries=20):
        # Connect to the Carla simulator
        for _ in range(tries):
            try:
                client = carla.Client('localhost', self.port)
                client.set_timeout(30)
                self.set_synchronous_mode(client)
                return
            except:
                time.sleep(self.sleep)
        logger.info(f'Failed to set synchronous mode {self.port}')


