import asyncio
from bleak import *

connected = False
i = 0

async def main():
    stop_event = asyncio.Event()

    def disconnect_cb(client):
        connected = False
        print('Disconnected')

    async def connect_cb(device: BLEDevice, advertising_data: AdvertisementData):
        global connected
        if device.name == 'PACController':
            eve = asyncio.Event()
            connected = True
            async with BleakClient(device, disconnected_callback=disconnect_cb) as connection:
                print(f'Connected to {connection.address}')
                # print(list([c.uuid for s in connection.services for c in s.characteristics ]))
                # 0xf1, 0xe3, 0xd5, 0xc7, 0xb9, 0xab, 0x9d, 0x8f,
                # 0x71, 0x63, 0x55, 0x47, 0x39, 0x2b, 0x1d, 0x0f
                # print(await connection.read_gatt_char('0f1d2b39-4755-6371-8f9d-abb9c7d5e3f1'))
                
                def recv(sender: BleakGATTCharacteristic, data: bytearray):
                    global i
                    print(f'#{i} ({sender.uuid}): {data}')
                    i += 1

                # while 1:
                #     print(await connection.read_gatt_char('0f1d2b39-4755-6371-8f9d-abb9c7d5e3f1'))



                # positions
                await connection.start_notify('0f1d2b39-4755-6371-8f9d-abb9c7d5e3f1', recv)

                # button
                await connection.start_notify('ffeddbc9-b7a5-9381-7f6d-5b4937251301', recv)
                print('yes')
                await eve.wait()
                pass
        pass

    async with BleakScanner(connect_cb) as scanner:
        # Important! Wait for an event to trigger stop, otherwise scanner
        # will stop immediately.
        await stop_event.wait()

    # scanner stops when block exits

if __name__ == '__main__':  
    asyncio.run(main())
