import React from 'react';

export interface ModelConfig {
    name: string;
    modelPath: string;
    classesPath: string;
    imgsz: number
}

interface ModelSelectorProps {
    models: ModelConfig[];
    selectedModel: ModelConfig;
    onModelSelect: (model: ModelConfig) => void;
}

const ModelSelector: React.FC<ModelSelectorProps> = ({
    models,
    selectedModel,
    onModelSelect,
}) => {
    return (
        <div style={{ marginBottom: '1rem' }}>
            <label style={{ marginRight: '0.5rem' }}>Select Model: </label>
            <select
                value={selectedModel.name}
                onChange={(e) => {
                    const selected = models.find((m) => m.name === e.target.value);
                    if (selected) onModelSelect(selected);
                }}
                style={{
                    padding: '0.5rem',
                    borderRadius: '4px',
                    border: '1px solid #ccc',
                    fontSize: '1rem',
                }}
            >
                {models.map((model) => (
                    <option key={model.name} value={model.name}>
                        {model.name}
                    </option>
                ))}
            </select>
        </div>
    );
};

export default ModelSelector; 