import React, { useEffect, useState } from 'react';

export interface CameraDevice {
    deviceId: string;
    label: string;
}

interface CameraSelectorProps {
    selectedDevice: CameraDevice | null;
    onDeviceSelect: (device: CameraDevice) => void;
}

const CameraSelector: React.FC<CameraSelectorProps> = ({
    selectedDevice,
    onDeviceSelect,
}) => {
    const [devices, setDevices] = useState<CameraDevice[]>([]);

    useEffect(() => {
        const loadDevices = async () => {
            try {
                await navigator.mediaDevices.getUserMedia({
                    video: {
                        width: { ideal: 1200 },
                        height: { ideal: 680 }
                    }
                });

                const allDevices = await navigator.mediaDevices.enumerateDevices();
                const videoDevices = allDevices
                    .filter(device => device.kind === 'videoinput')
                    .map(device => ({
                        deviceId: device.deviceId,
                        label: device.label || `Camera ${device.deviceId.slice(0, 5)}...`
                    }));

                setDevices(videoDevices);

                if (!selectedDevice && videoDevices.length > 0) {
                    onDeviceSelect(videoDevices[0]);
                }
            } catch (error) {
                console.error('Error accessing camera devices:', error);
            }
        };

        loadDevices();
    }, []);

    return (
        <div style={{ marginBottom: '1rem', marginLeft: '1rem', display: 'inline-block' }}>
            <label style={{ marginRight: '0.5rem' }}>Select Camera: </label>
            <select
                value={selectedDevice?.deviceId || ''}
                onChange={(e) => {
                    const device = devices.find(d => d.deviceId === e.target.value);
                    if (device) onDeviceSelect(device);
                }}
                style={{
                    padding: '0.5rem',
                    borderRadius: '4px',
                    border: '1px solid #ccc',
                    fontSize: '1rem',
                }}
            >
                {devices.map((device) => (
                    <option key={device.deviceId} value={device.deviceId}>
                        {device.label}
                    </option>
                ))}
            </select>
        </div>
    );
};

export default CameraSelector; 