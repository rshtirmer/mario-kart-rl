-- Super Mario Kart reward and done logic for stable-retro
-- Checkpoint-based progress reward; done on race finish or mode change

prev_progress = 0
step_count = 0
wall_hits = 0

function getReward()
    -- Cumulative checkpoint progress across laps
    local lap_num = data.lap - 128
    if lap_num < 0 then lap_num = 0 end
    local total_cp = data.total_checkpoints
    if total_cp == 0 then total_cp = 1 end

    local progress = data.checkpoint + lap_num * total_cp

    local reward = progress - prev_progress

    -- Sanity check: large negative means state reset, not regression
    if reward < -100 then
        reward = 0
    end

    prev_progress = progress
    step_count = step_count + 1

    return reward
end

function isDone()
    -- Not in racing mode
    if data.ext_mode ~= 28 then
        return true
    end
    -- Completed 5 laps (lap field = 128 + lap_number)
    if data.lap >= 133 then
        return true
    end
    return false
end
